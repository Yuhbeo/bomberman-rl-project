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
               e.KILLED_SELF: -40.0, e.GOT_KILLED: -40.0,
               e.KILLED_OPPONENT: 20.0, e.INVALID_ACTION: -5.0}
    # No positive escape/bomb-drop rewards that can be farmed in a cycle.
    return -0.1 + sum(rewards.get(event, 0.0) for event in events)


def navigation_reward(old, action):
    state = old[0]
    kind, direction, _, danger = state[:4]
    if kind and direction < 4 and not danger:
        # Equal positive/negative magnitudes: a there-and-back cycle loses
        # the per-step cost. No navigation reward while escaping a bomb.
        return 1.0 if ACTIONS.index(action) == direction else -1.0
    return 0.0


def update(self, old, action, reward, new=None):
    state, _, phi = old
    values = self.q_table.setdefault(state, np.zeros(len(ACTIONS)))
    target = reward + navigation_reward(old, action) - phi
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
        _, old, action, reward, new = self.pending
        update(self, old, action, reward, new)
        self.pending = None
    if old_game_state is None:
        return
    old = old_observation(self, old_game_state)
    new = observe(new_game_state) if new_game_state is not None else None
    key = (old_game_state['round'], old_game_state['step'])
    # Delay one update: the engine sends surviving final transitions twice
    # (game_events_occurred and end_of_round), without an explicit done flag.
    self.pending = (key, old, self_action, reward_from_events(events), new)


def end_of_round(self, last_game_state, last_action, events):
    key = None if last_game_state is None else (last_game_state['round'], last_game_state['step'])
    terminal_old = None
    if self.pending is not None:
        pending_key, old, action, reward, new = self.pending
        if pending_key == key:
            terminal_old = old
        else:
            update(self, old, action, reward, new)
        self.pending = None
    if last_game_state is not None and last_action is not None:
        old = terminal_old if terminal_old is not None else old_observation(self, last_game_state)
        update(self, old, last_action, reward_from_events(events))
    self.training_rounds += 1
    payload = {'version': MODEL_VERSION, 'training_rounds': self.training_rounds,
               'q_table': self.q_table}
    temporary = MODEL_FILE + '.tmp'
    with open(temporary, 'wb') as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, MODEL_FILE)
