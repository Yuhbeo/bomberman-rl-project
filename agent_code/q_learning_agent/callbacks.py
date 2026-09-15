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
        try:
            self.logger.info("Loading Q-table.")

            with open(MODEL_FILE, "rb") as file:
                self.q_table = pickle.load(file)

        except (EOFError, pickle.UnpicklingError):
            self.logger.warning(
                "Q-table file is empty or corrupted. Creating new Q-table."
            )
            self.q_table = {}

    else:
        self.logger.info("Creating new Q-table.")
        self.q_table = {}


def act(self, game_state: dict) -> str:
    """
    Wählt eine Aktion mit epsilon-greedy Q-Learning.
    """

    state = state_to_features(game_state)

    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))

    epsilon = 0.20

    # State:
    # 0 coin_direction
    # 1 crate_direction
    # 2 up_free
    # 3 right_free
    # 4 down_free
    # 5 left_free
    # 6 in_danger
    # 7 escape_direction
    # 8 bomb_available
    # 9 crate_nearby

    in_danger = state[6]
    bomb_available = state[8]
    crate_nearby = state[9]

    # Alle vier Bewegungsaktionen bleiben möglich.
    allowed_action_indices = [0, 1, 2, 3]

    # Bombe nur, wenn sie verfügbar ist,
    # eine Kiste direkt daneben steht
    # und wir aktuell nicht in Gefahr sind.
    if (
        bomb_available
        and crate_nearby
        and not in_danger
    ):
        allowed_action_indices.append(5)

    # Exploration
    if self.train and random.random() < epsilon:
        action_index = random.choice(
            allowed_action_indices
        )
        return ACTIONS[action_index]

    # Exploitation
    q_values = self.q_table[state][
        allowed_action_indices
    ]

    max_q = np.max(q_values)

    best_positions = np.where(
        q_values == max_q
    )[0]

    selected_position = random.choice(
        best_positions
    )

    action_index = allowed_action_indices[
        selected_position
    ]

    return ACTIONS[action_index]


def direction_to_nearest_coin(
    field,
    start,
    coins
):
    """
    BFS zum nächsten erreichbaren Coin.
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

        if (x, y) in coins:
            return first_action

        for (dx, dy), action in directions:

            nx = x + dx
            ny = y + dy

            next_position = (
                nx,
                ny
            )

            if next_position in visited:
                continue

            if field[nx, ny] != 0:
                continue

            visited.add(
                next_position
            )

            if first_action is None:
                new_first_action = action
            else:
                new_first_action = first_action

            queue.append(
                (
                    next_position,
                    new_first_action
                )
            )

    return None


def direction_to_nearest_crate(
    field,
    start
):
    """
    BFS zu einem freien Feld direkt neben der nächsten Kiste.
    """

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

        # Kiste direkt neben diesem Feld?
        for (dx, dy), _ in directions:

            nx = x + dx
            ny = y + dy

            if field[nx, ny] == 1:
                return first_action

        # BFS fortsetzen
        for (dx, dy), action in directions:

            nx = x + dx
            ny = y + dy

            next_position = (
                nx,
                ny
            )

            if next_position in visited:
                continue

            if field[nx, ny] != 0:
                continue

            visited.add(
                next_position
            )

            if first_action is None:
                new_first_action = action
            else:
                new_first_action = first_action

            queue.append(
                (
                    next_position,
                    new_first_action
                )
            )

    return None


def get_danger_map(game_state):
    """
    Binäre Gefahrenkarte.

    0 = sicher
    1 = Bombe / Explosion bedroht dieses Feld
    """

    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    danger_map = np.zeros_like(
        field,
        dtype=int
    )

    # Aktive Explosionen
    danger_map[
        explosion_map > 0
    ] = 1

    directions = [
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0)
    ]

    for (bomb_x, bomb_y), timer in bombs:

        danger_map[
            bomb_x,
            bomb_y
        ] = 1

        for dx, dy in directions:

            for distance in range(1, 4):

                x = bomb_x + dx * distance
                y = bomb_y + dy * distance

                if (
                    x < 0
                    or x >= field.shape[0]
                    or y < 0
                    or y >= field.shape[1]
                ):
                    break

                # Steinwand stoppt Explosion.
                if field[x, y] == -1:
                    break

                danger_map[x, y] = 1

                # Kiste wird getroffen und stoppt Explosion.
                if field[x, y] == 1:
                    break

    return danger_map


def direction_to_safety(game_state):
    """
    BFS zum nächsten sicheren Feld.
    """

    field = game_state["field"]

    x, y = game_state["self"][3]

    danger_map = get_danger_map(
        game_state
    )

    bomb_positions = {
        position
        for position, timer
        in game_state["bombs"]
    }

    start = (
        x,
        y
    )

    # Bereits sicher.
    if danger_map[x, y] == 0:
        return None

    queue = deque()
    queue.append(
        (start, None)
    )

    visited = {
        start
    }

    directions = [
        ((0, -1), "UP"),
        ((1, 0), "RIGHT"),
        ((0, 1), "DOWN"),
        ((-1, 0), "LEFT")
    ]

    while queue:

        (
            current_x,
            current_y
        ), first_action = queue.popleft()

        # Sicheres Feld gefunden.
        if danger_map[
            current_x,
            current_y
        ] == 0:

            return first_action

        for (dx, dy), action in directions:

            nx = current_x + dx
            ny = current_y + dy

            next_position = (
                nx,
                ny
            )

            if next_position in visited:
                continue

            if field[nx, ny] != 0:
                continue

            if next_position in bomb_positions:
                continue

            visited.add(
                next_position
            )

            if first_action is None:
                new_first_action = action
            else:
                new_first_action = first_action

            queue.append(
                (
                    next_position,
                    new_first_action
                )
            )

    return None


def state_to_features(game_state: dict):
    """
    Kompakter State für die Q-Tabelle.

    0 coin_direction
    1 crate_direction
    2 up_free
    3 right_free
    4 down_free
    5 left_free
    6 in_danger
    7 escape_direction
    8 bomb_available
    9 crate_nearby
    """

    if game_state is None:
        return None

    x, y = game_state[
        "self"
    ][3]

    field = game_state[
        "field"
    ]

    coins = game_state[
        "coins"
    ]

    danger_map = get_danger_map(
        game_state
    )

    coin_direction = direction_to_nearest_coin(
        field,
        (x, y),
        coins
    )

    crate_direction = direction_to_nearest_crate(
        field,
        (x, y)
    )

    # Genau wie beim ursprünglichen Agenten:
    # nur das field wird hier betrachtet.
    up_free = int(
        field[x, y - 1] == 0
    )

    right_free = int(
        field[x + 1, y] == 0
    )

    down_free = int(
        field[x, y + 1] == 0
    )

    left_free = int(
        field[x - 1, y] == 0
    )

    crate_nearby = int(
        field[x, y - 1] == 1
        or field[x + 1, y] == 1
        or field[x, y + 1] == 1
        or field[x - 1, y] == 1
    )

    in_danger = int(
        danger_map[x, y] == 1
    )

    escape_direction = direction_to_safety(
        game_state
    )

    bomb_available = int(
        game_state["self"][2]
    )

    state = (
        coin_direction,
        crate_direction,

        up_free,
        right_free,
        down_free,
        left_free,

        in_danger,
        escape_direction,

        bomb_available,
        crate_nearby
    )

    return state