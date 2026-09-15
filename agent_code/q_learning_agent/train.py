"""Q-learning with potential shaping and exactly one update per transition."""

import os
import pickle

import numpy as np

import events as e
from .callbacks import ACTIONS, MODEL_FILE, MODEL_VERSION, observe

ALPHA = 0.15
GAMMA = 0.95


def setup_training(self):
    self.pending = None


def reward_from_events(events):
    rewards = {e.COIN_COLLECTED: 10.0, e.CRATE_DESTROYED: 3.0,
               e.KILLED_OPPONENT: 40.0, e.INVALID_ACTION: -5.0}
    reward = -0.1 + sum(rewards.get(event, 0.0) for event in events)
    # A self-kill emits BOTH death events; count death exactly once.
    if e.GOT_KILLED in events or e.KILLED_SELF in events:
        reward -= 60.0
    return reward


def focus_opponent(state):
    kind, _, mask, danger = state[:4]
    direction, distance = state[7:9]
    return (not danger and kind != 1 and direction < 4
            and bool(mask & (1 << direction))
            and (kind == 0 or distance <= 3))


def navigation_reward(old, action):
    state = old[0]
    kind, direction, _, danger = state[:4]
    if kind and direction < 4 and not danger and not focus_opponent(state):
        return 1.0 if ACTIONS.index(action) == direction else -1.0
    return 0.0


def opponent_navigation_reward(old, action):
    state = old[0]
    if not focus_opponent(state):
        return 0.0
    # Do not punish firing at an opponent already in range.
    if action == 'BOMB' and state[9]:
        return 0.0
    return 0.6 if ACTIONS.index(action) == state[7] else -0.6


def opponent_bomb_reward(old, action, bomb_dropped):
    if action != 'BOMB' or not bomb_dropped:
        return 0.0
    state = old[0]
    # A predicted trap is not an actual kill. Keep this shaping bonus modest.
    return 2.0 if state[10] else (1.0 if state[9] else 0.0)


def update(self, old, action, reward, new=None, bomb_dropped=False):
    state, _, phi = old
    values = self.q_table.setdefault(state, np.zeros(len(ACTIONS)))
    target = (reward + navigation_reward(old, action)
              + opponent_navigation_reward(old, action)
              + opponent_bomb_reward(old, action, bomb_dropped) - phi)
    if new is not None:
        next_state, allowed, next_phi = new
        next_values = self.q_table.setdefault(next_state, np.zeros(len(ACTIONS)))
        target += GAMMA * (next_phi + np.max(next_values[allowed]))
    i = ACTIONS.index(action)
    values[i] += ALPHA * (target - values[i])


def old_observation(self, game_state):
    key = (game_state['round'], game_state['step'])
    if getattr(self, 'last_observation_key', None) == key:
        return self.last_observation
    return observe(game_state)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if self.pending is not None:
        _, old, action, reward, new, dropped = self.pending
        update(self, old, action, reward, new, dropped)
        self.pending = None
    if old_game_state is None:
        return
    old = old_observation(self, old_game_state)
    new = observe(new_game_state) if new_game_state is not None else None
    key = (old_game_state['round'], old_game_state['step'])
    # Delay one update: the engine sends surviving final transitions twice
    # (game_events_occurred and end_of_round), without an explicit done flag.
    self.pending = (key, old, self_action, reward_from_events(events), new, e.BOMB_DROPPED in events)


def end_of_round(self, last_game_state, last_action, events):
    key = None if last_game_state is None else (last_game_state['round'], last_game_state['step'])
    terminal_old = None
    if self.pending is not None:
        pending_key, old, action, reward, new, dropped = self.pending
        if pending_key == key:
            terminal_old = old
        else:
            update(self, old, action, reward, new, dropped)
        self.pending = None
    if last_game_state is not None and last_action is not None:
        old = terminal_old if terminal_old is not None else old_observation(self, last_game_state)
        update(self, old, last_action, reward_from_events(events),
               bomb_dropped=e.BOMB_DROPPED in events)
    self.training_rounds += 1
    self.session_rounds += 1
    payload = {'version': MODEL_VERSION, 'training_rounds': self.training_rounds,
               'q_table': self.q_table}
    temporary = MODEL_FILE + '.tmp'
    with open(temporary, 'wb') as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, MODEL_FILE)
