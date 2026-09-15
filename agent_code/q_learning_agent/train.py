import pickle

import numpy as np

import events as e

from .callbacks import (
    state_to_features,
    ACTIONS,
    MODEL_FILE
)


ALPHA = 0.1
GAMMA = 0.9


# =============================================================
# CUSTOM EVENTS
# =============================================================

MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_TOWARDS_CRATE = "MOVED_TOWARDS_CRATE"
NOT_TOWARDS_TARGET = "NOT_TOWARDS_TARGET"

FOLLOWED_ESCAPE_DIRECTION = "FOLLOWED_ESCAPE_DIRECTION"
IGNORED_ESCAPE_DIRECTION = "IGNORED_ESCAPE_DIRECTION"
ESCAPED_DANGER = "ESCAPED_DANGER"
ENTERED_DANGER = "ENTERED_DANGER"

BOMB_NEAR_CRATE = "BOMB_NEAR_CRATE"
BOMB_WITHOUT_CRATE = "BOMB_WITHOUT_CRATE"

BACKTRACKED = "BACKTRACKED"


def setup_training(self):
    """
    Wird einmal am Anfang des Trainings aufgerufen.
    """

    self.position_history = []


def game_events_occurred(
    self,
    old_game_state,
    self_action,
    new_game_state,
    events
):
    """
    Wird nach jedem Spielschritt aufgerufen.
    """

    if old_game_state is None:
        return

    # ---------------------------------------------------------
    # States
    # ---------------------------------------------------------

    old_state = state_to_features(
        old_game_state
    )

    new_state = state_to_features(
        new_game_state
    )

    old_position = old_game_state[
        "self"
    ][3]

    new_position = new_game_state[
        "self"
    ][3]

    # ---------------------------------------------------------
    # Neue Q-States
    # ---------------------------------------------------------

    if old_state not in self.q_table:
        self.q_table[
            old_state
        ] = np.zeros(
            len(ACTIONS)
        )

    if new_state not in self.q_table:
        self.q_table[
            new_state
        ] = np.zeros(
            len(ACTIONS)
        )

    # ---------------------------------------------------------
    # Features
    # ---------------------------------------------------------

    coin_direction = old_state[0]
    crate_direction = old_state[1]

    old_in_danger = old_state[6]
    escape_direction = old_state[7]

    crate_nearby = old_state[9]

    new_in_danger = new_state[6]

    # =========================================================
    # BACKTRACKING
    # =========================================================

    if len(self.position_history) == 0:
        self.position_history.append(
            old_position
        )

    if (
        not old_in_danger
        and len(self.position_history) >= 2
        and new_position
        == self.position_history[-2]
    ):
        events.append(
            BACKTRACKED
        )

    self.position_history.append(
        new_position
    )

    if len(self.position_history) > 4:
        self.position_history.pop(
            0
        )

    # =========================================================
    # ESCAPE / DANGER
    # =========================================================

    if old_in_danger:

        if escape_direction is not None:

            if self_action == escape_direction:

                events.append(
                    FOLLOWED_ESCAPE_DIRECTION
                )

            elif self_action in [
                "UP",
                "RIGHT",
                "DOWN",
                "LEFT"
            ]:

                events.append(
                    IGNORED_ESCAPE_DIRECTION
                )

        if new_in_danger == 0:

            events.append(
                ESCAPED_DANGER
            )

    else:

        # Freiwillig in Gefahrenzone gelaufen.
        # BOMB wird ausgenommen.
        if (
            new_in_danger == 1
            and self_action != "BOMB"
        ):

            events.append(
                ENTERED_DANGER
            )

        # -----------------------------------------------------
        # Richtung Coin
        # -----------------------------------------------------

        if coin_direction is not None:

            if self_action == coin_direction:

                events.append(
                    MOVED_TOWARDS_COIN
                )

            elif self_action in [
                "UP",
                "RIGHT",
                "DOWN",
                "LEFT"
            ]:

                events.append(
                    NOT_TOWARDS_TARGET
                )

        # -----------------------------------------------------
        # Richtung Kiste
        # -----------------------------------------------------

        elif crate_direction is not None:

            if self_action == crate_direction:

                events.append(
                    MOVED_TOWARDS_CRATE
                )

            elif self_action in [
                "UP",
                "RIGHT",
                "DOWN",
                "LEFT"
            ]:

                events.append(
                    NOT_TOWARDS_TARGET
                )

    # =========================================================
    # BOMB EVENTS
    # =========================================================

    if self_action == "BOMB":

        if crate_nearby:

            events.append(
                BOMB_NEAR_CRATE
            )

        else:

            events.append(
                BOMB_WITHOUT_CRATE
            )

    # =========================================================
    # Q-LEARNING
    # =========================================================

    action_index = ACTIONS.index(
        self_action
    )

    reward = reward_from_events(
        events
    )

    old_q = self.q_table[
        old_state
    ][
        action_index
    ]

    # ---------------------------------------------------------
    # Mögliche Aktionen im nächsten State
    # ---------------------------------------------------------

    allowed_action_indices = [
        0,
        1,
        2,
        3
    ]

    new_bomb_available = new_state[8]
    new_crate_nearby = new_state[9]

    if (
        new_bomb_available
        and new_crate_nearby
        and not new_in_danger
    ):
        allowed_action_indices.append(
            5
        )

    best_next_q = np.max(
        self.q_table[
            new_state
        ][
            allowed_action_indices
        ]
    )

    target = (
        reward
        + GAMMA * best_next_q
    )

    new_q = (
        old_q
        + ALPHA
        * (
            target
            - old_q
        )
    )

    self.q_table[
        old_state
    ][
        action_index
    ] = new_q


def end_of_round(
    self,
    last_game_state,
    last_action,
    events
):
    """
    Terminales Q-Update und Q-Tabelle speichern.
    """

    if (
        last_game_state is not None
        and last_action is not None
    ):

        state = state_to_features(
            last_game_state
        )

        if state not in self.q_table:

            self.q_table[
                state
            ] = np.zeros(
                len(ACTIONS)
            )

        action_index = ACTIONS.index(
            last_action
        )

        reward = reward_from_events(
            events
        )

        old_q = self.q_table[
            state
        ][
            action_index
        ]

        # Terminal:
        # kein Bootstrap mehr.
        new_q = (
            old_q
            + ALPHA
            * (
                reward
                - old_q
            )
        )

        self.q_table[
            state
        ][
            action_index
        ] = new_q

    self.position_history = []

    with open(
        MODEL_FILE,
        "wb"
    ) as file:

        pickle.dump(
            self.q_table,
            file
        )


def reward_from_events(events):
    """
    Reward-Summe des aktuellen Schritts.
    """

    rewards = {

        # Hauptziele
        e.COIN_COLLECTED: 10,
        e.CRATE_DESTROYED: 3,
        e.COIN_FOUND: 1,

        # Tod
        e.KILLED_SELF: -20,
        e.GOT_KILLED: -20,

        # Fehler / Warten
        e.INVALID_ACTION: -2,
        e.WAITED: -1,

        # Navigation
        MOVED_TOWARDS_COIN: 0.5,
        MOVED_TOWARDS_CRATE: 0.3,
        NOT_TOWARDS_TARGET: -0.2,

        BACKTRACKED: -1.0,

        # Gefahr / Flucht
        FOLLOWED_ESCAPE_DIRECTION: 1.0,
        IGNORED_ESCAPE_DIRECTION: -1.0,

        ESCAPED_DANGER: 3,
        ENTERED_DANGER: -2,

        # Bomben
        BOMB_NEAR_CRATE: 1,
        BOMB_WITHOUT_CRATE: -1,
    }

    reward_sum = 0

    for event in events:

        if event in rewards:

            reward_sum += rewards[
                event
            ]

    return reward_sum