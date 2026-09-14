import os
import pickle
import random
from collections import deque

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = "q_table.pkl"


def setup(self):
    """
    Wird einmal beim Start des Agents aufgerufen.
    """

    if os.path.isfile(MODEL_FILE):
        self.logger.info("Loading Q-table.")
        with open(MODEL_FILE, "rb") as file:
            self.q_table = pickle.load(file)
    else:
        self.logger.info("Creating new Q-table.")
        self.q_table = {}


def act(self, game_state: dict) -> str:
    state = state_to_features(game_state)

    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))

    epsilon = 0.10

    # Exploration
    if self.train and random.random() < epsilon:
        return random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT'])

    # Für Task 1 nur Bewegungsaktionen berücksichtigen
    movement_q_values = self.q_table[state][:4]

    max_q = np.max(movement_q_values)

    best_actions = np.where(movement_q_values == max_q)[0]

    action_index = random.choice(best_actions)

    return ACTIONS[action_index]

def direction_to_nearest_coin(field, start, coins):
    """
    Finds a shortest path to the nearest reachable coin using BFS.
    Returns only the first movement direction of that path.

    Possible return values:
    "UP", "RIGHT", "DOWN", "LEFT", None
    """

    if not coins:
        return None

    queue = deque()
    queue.append((start, None))

    visited = {start}

    directions = [
        ((0, -1), "UP"),
        ((1, 0), "RIGHT"),
        ((0, 1), "DOWN"),
        ((-1, 0), "LEFT")
    ]

    while queue:
        (x, y), first_action = queue.popleft()

        # Coin reached
        if (x, y) in coins:
            return first_action

        for (dx, dy), action in directions:
            nx = x + dx
            ny = y + dy
            next_position = (nx, ny)

            # Already checked
            if next_position in visited:
                continue

            # Only walk on free tiles
            if field[nx, ny] != 0:
                continue

            visited.add(next_position)

            # Remember the first action of the path
            if first_action is None:
                new_first_action = action
            else:
                new_first_action = first_action

            queue.append((next_position, new_first_action))

    # No reachable coin found
    return None

def get_danger_map(game_state):
    """
    Creates a binary danger map.

    0 = currently safe
    1 = threatened by a bomb or active explosion
    """

    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    danger_map = np.zeros_like(field, dtype=int)

    # Tiles with an active explosion are dangerous
    danger_map[explosion_map > 0] = 1

    for (bomb_x, bomb_y), timer in bombs:

        danger_map[bomb_x, bomb_y] = 1

        directions = [
            (0, -1),   # UP
            (1, 0),    # RIGHT
            (0, 1),    # DOWN
            (-1, 0)    # LEFT
        ]

        for dx, dy in directions:

            for distance in range(1, 4):

                x = bomb_x + dx * distance
                y = bomb_y + dy * distance

                # Stay inside board
                if (
                    x < 0 or x >= field.shape[0]
                    or y < 0 or y >= field.shape[1]
                ):
                    break

                # Stone wall blocks explosion
                if field[x, y] == -1:
                    break

                danger_map[x, y] = 1

                # Explosion hits crate but does not continue
                if field[x, y] == 1:
                    break

    return danger_map

def direction_to_safety(game_state):
    """
    Finds a shortest path from the agent's current position
    to a safe tile.

    Returns:
    "UP", "RIGHT", "DOWN", "LEFT" or None
    """

    field = game_state["field"]
    x, y = game_state["self"][3]

    danger_map = get_danger_map(game_state)
    
    bomb_positions = {
    position for position, timer in game_state["bombs"]
}

    start = (x, y)

    # Already safe
    if danger_map[x, y] == 0:
        return None

    queue = deque()
    queue.append((start, None))

    visited = {start}

    directions = [
        ((0, -1), "UP"),
        ((1, 0), "RIGHT"),
        ((0, 1), "DOWN"),
        ((-1, 0), "LEFT")
    ]

    while queue:
        (current_x, current_y), first_action = queue.popleft()

        # Safe tile found
        if danger_map[current_x, current_y] == 0:
            return first_action

        for (dx, dy), action in directions:
            nx = current_x + dx
            ny = current_y + dy

            next_position = (nx, ny)

            if next_position in visited:
                continue

            # Only free tiles can be entered
            if field[nx, ny] != 0:
                continue

            if next_position in bomb_positions:
                continue

            visited.add(next_position)

            if first_action is None:
                new_first_action = action
            else:
                new_first_action = first_action

            queue.append((next_position, new_first_action))

    # No escape found
    return None

def state_to_features(game_state: dict):
    if game_state is None:
        return None

    x, y = game_state["self"][3]
    
    danger_map = get_danger_map(game_state)

    if game_state["bombs"]:
        print("Position:", (x, y))
        print("Bombs:", game_state["bombs"])
        print("Danger:", danger_map[x, y])
        print("Escape:", direction_to_safety(game_state))
        print()

    field = game_state["field"]
    coins = game_state["coins"]

    coin_direction = direction_to_nearest_coin(
        field,
        (x, y),
        coins
    )

    # Prüfen, ob die vier Nachbarfelder frei sind.
    # field[x, y] == 0 bedeutet: begehbares Feld.
    up_free = int(field[x, y - 1] == 0)
    right_free = int(field[x + 1, y] == 0)
    down_free = int(field[x, y + 1] == 0)
    left_free = int(field[x - 1, y] == 0)
    
    in_danger = int(danger_map[x, y] == 1)

    escape_direction = direction_to_safety(game_state)

    state = (
        coin_direction,
        up_free,
        right_free,
        down_free,
        left_free,
        in_danger,
        escape_direction,
    )

    return state
