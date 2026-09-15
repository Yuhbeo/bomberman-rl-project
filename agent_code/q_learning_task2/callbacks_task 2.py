import os
import pickle
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

import numpy as np
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))
MODEL_FILE = str(Path(__file__).with_name('q_table.pkl'))
MODEL_VERSION = 3


def setup(self):
    self.q_table = {}
    self.training_rounds = 0
    if os.path.isfile(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as stream:
            model = pickle.load(stream)
        if model.get('version') != MODEL_VERSION:
            raise ValueError('Incompatible Q-table. Use the task2_v3 model or retrain.')
        self.q_table = model['q_table']
        self.training_rounds = model['training_rounds']
    self.rng = np.random.default_rng(int(os.environ.get('BOMBERMAN_AGENT_SEED', '12345')))
    self.history = deque(maxlen=16)
    self.round_id = None
    self.loop_breaks = 0


def blast_tiles(field, origin):
    """Match items.Bomb: ONLY stone walls stop a blast, not crates/bombs."""
    result = [origin]
    for dx, dy in DELTAS[:4]:
        for distance in range(1, s.BOMB_POWER + 1):
            p = (origin[0] + dx * distance, origin[1] + dy * distance)
            if not inside(field, p) or field[p] == -1:
                break
            result.append(p)
    return result


def inside(field, p):
    return 0 <= p[0] < field.shape[0] and 0 <= p[1] < field.shape[1]


def safety_analysis(game_state, plant=False):
    """Return safe first moves and a future danger union.

    Time 1 is AFTER the next action. Timer 0 detonates at time 1.
    A newly planted timer-4 bomb detonates at time 5, not time 4.
    The upstream engine has no chain reactions. Crates and opponents are
    conservatively kept blocked over the short search horizon. Future opponent
    actions are unknown; this is not a guarantee against adversarial opponents.
    """
    field = game_state['field']
    start = game_state['self'][3]
    others = {a[3] for a in game_state['others']}
    bombs = list(game_state['bombs'])
    if plant:
        bombs.append((start, s.BOMB_TIMER))
    detonation = {p: max(0, int(timer)) + 1 for p, timer in bombs}
    horizon = max([1, int(np.max(game_state['explosion_map']))] +
                  [t + s.EXPLOSION_TIMER - 1 for t in detonation.values()])
    danger = np.zeros((horizon + 1,) + field.shape, dtype=bool)
    for t in range(1, horizon + 1):
        # explosion_map already accounts for the next update_explosions().
        danger[t] = game_state['explosion_map'] >= t
    for p, det in detonation.items():
        for tile in blast_tiles(field, p):
            danger[det:det + s.EXPLOSION_TIMER, tile[0], tile[1]] = True
    future = np.logical_or.accumulate(danger[::-1], axis=0)[::-1]

    def legal(p, q, t):
        return (inside(field, q) and field[q] == 0 and q not in others
                and (q == p or q not in detonation or t > detonation[q]))

    @lru_cache(None)
    def survives(p, t):
        if danger[t, p[0], p[1]]:
            return False
        if t == horizon or not future[t, p[0], p[1]]:
            return True  # staying here is safe until all known flames expire
        for dx, dy in DELTAS:
            q = (p[0] + dx, p[1] + dy)
            if legal(p, q, t + 1) and survives(q, t + 1):
                return True
        return False

    safe = []
    for i, (dx, dy) in enumerate(DELTAS):
        p = (start[0] + dx, start[1] + dy)
        if legal(start, p, 1) and survives(p, 1):
            safe.append(i)
    return safe, future[1]


def target_info(game_state):
    """BFS to coins first, otherwise a reachable tile adjacent to a crate.

    Bombs, opponents and currently dangerous flames block navigation.
    Return kind, first direction, distance. No action is prescribed by this
    feature: Q-learning assigns values to all safe actions.
    """
    field = game_state['field']
    start = game_state['self'][3]
    blocked = {p for p, _ in game_state['bombs']} | {a[3] for a in game_state['others']}
    coins = set(game_state['coins'])
    queue = deque([(start, 4, 0)])
    seen = {start}
    crate_target = None
    while queue:
        p, first, distance = queue.popleft()
        if p in coins:
            return 1, first, distance
        if crate_target is None and any(
                inside(field, (p[0] + dx, p[1] + dy)) and
                field[p[0] + dx, p[1] + dy] == 1 for dx, dy in DELTAS[:4]):
            crate_target = (2, first, distance)
        for i, (dx, dy) in enumerate(DELTAS[:4]):
            q = (p[0] + dx, p[1] + dy)
            if (q not in seen and inside(field, q) and field[q] == 0
                    and q not in blocked and game_state['explosion_map'][q] == 0):
                seen.add(q)
                queue.append((q, i if distance == 0 else first, distance + 1))
    return crate_target or (0, 4, 0)


def observe(game_state):
    field = game_state['field']
    pos = game_state['self'][3]
    safe, danger = safety_analysis(game_state)
    useful = any(field[p] == 1 for p in blast_tiles(field, pos))
    if (game_state['self'][2] and useful and
            pos not in {p for p, _ in game_state['bombs']}):
        after_plant, _ = safety_analysis(game_state, plant=True)
        if 4 in after_plant:  # BOMB consumes one step without moving!
            safe.append(5)
    if not safe:
        # Already trapped: do not crash or introduce invalid movements.
        blocked = {p for p, _ in game_state['bombs']} | {a[3] for a in game_state['others']}
        safe = [i for i, (dx, dy) in enumerate(DELTAS[:4])
                if inside(field, (pos[0] + dx, pos[1] + dy))
                and field[pos[0] + dx, pos[1] + dy] == 0
                and (pos[0] + dx, pos[1] + dy) not in blocked] + [4]
    kind, direction, distance = target_info(game_state)
    mask = sum(1 << i for i in safe)
    features = (kind, direction, mask, int(danger[pos]),
                int(game_state['self'][2]), int(useful),
                min(distance, 4))
    # Bounded, state-only potential. No repeated reward for leaving danger.
    potential = (6.0 if kind == 1 else 2.0) / (distance + 1) if kind else 0.0
    return features, safe, potential


def state_to_features(game_state):
    return None if game_state is None else observe(game_state)[0]


def act(self, game_state):
    if self.round_id != game_state['round']:
        self.history.clear()
        self.round_id = game_state['round']
    pos = game_state['self'][3]
    self.history.append(pos)
    observation = observe(game_state)
    self.last_observation = observation
    self.last_observation_key = (game_state['round'], game_state['step'])
    state, allowed, _ = observation
    values = self.q_table.setdefault(state, np.zeros(len(ACTIONS)))
    epsilon = max(0.03, 0.25 * (0.997 ** self.training_rounds)) if self.train else 0.0
    if self.rng.random() < epsilon:
        return ACTIONS[int(self.rng.choice(allowed))]
    scores = values[allowed].copy()
    # Explicit, small recovery heuristic for state aliasing, also active in
    # evaluation. It cannot disable the safety mask or punish necessary escapes.
    visits = Counter(self.history)
    if visits[pos] >= 3 and not state[3]:
        self.loop_breaks += 1
        for j, action in enumerate(allowed):
            if action < 5:
                dx, dy = DELTAS[action]
                scores[j] -= 2.0 * visits[(pos[0] + dx, pos[1] + dy)]
    best = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0, atol=1e-10))
    return ACTIONS[allowed[int(self.rng.choice(best))]]
