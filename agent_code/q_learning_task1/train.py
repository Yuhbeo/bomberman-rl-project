import pickle

import numpy as np
import events as e

from .callbacks import state_to_features, ACTIONS, MODEL_FILE


# Q-Learning Hyperparameter
ALPHA = 0.1
GAMMA = 0.9


# Eigene Events für Reward Shaping
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
NOT_TOWARDS_COIN = "NOT_TOWARDS_COIN"


def setup_training(self):
    """
    Wird einmal beim Start des Trainings aufgerufen.
    Für unsere einfache Q-Learning-Version müssen wir hier
    aktuell nichts zusätzlich initialisieren.
    """
    pass


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: list
):
    """
    Wird nach jedem Spielschritt aufgerufen.

    Hier:
    1. Zustände bestimmen
    2. zusätzliche Rewards erzeugen
    3. Reward berechnen
    4. Q-Wert aktualisieren
    """

    if old_game_state is None:
        return

    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    # Falls ein State noch nicht in der Q-Tabelle existiert:
    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))

    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(ACTIONS))

    # ---------------------------------------------------------
    # Reward Shaping:
    #
    # old_state[0] enthält die BFS-Richtung zum nächsten Coin.
    #
    # Beispiel:
    # old_state = ("RIGHT", 1, 1, 0, 1)
    #
    # Wenn der Agent RIGHT wählt, bewegt er sich in Richtung
    # des kürzesten Weges zum Coin.
    # ---------------------------------------------------------

    coin_direction = old_state[0]

    if coin_direction is not None:
        if self_action == coin_direction:
            events.append(MOVED_TOWARDS_COIN)
        else:
            events.append(NOT_TOWARDS_COIN)

    # Index der ausgeführten Aktion
    action_index = ACTIONS.index(self_action)

    # Reward aus allen Events bestimmen
    reward = reward_from_events(events)

    # Alter Q-Wert Q(s, a)
    old_q = self.q_table[old_state][action_index]

    # Bester zukünftiger Q-Wert.
    # Task 1 benutzt nur:
    # UP, RIGHT, DOWN, LEFT
    best_next_q = np.max(self.q_table[new_state][:4])

    # Q-Learning Formel:
    #
    # Q(s,a) <- Q(s,a)
    #           + alpha * (
    #               reward
    #               + gamma * max Q(s',a')
    #               - Q(s,a)
    #           )

    new_q = old_q + ALPHA * (
        reward
        + GAMMA * best_next_q
        - old_q
    )

    self.q_table[old_state][action_index] = new_q


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: list
):
    """
    Wird am Ende einer Runde aufgerufen.

    Beim letzten Schritt gibt es keinen Folgezustand mehr.
    Deshalb wird nur noch der finale Reward berücksichtigt.

    Anschließend wird die Q-Tabelle gespeichert.
    """

    if last_game_state is not None and last_action is not None:

        state = state_to_features(last_game_state)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))

        action_index = ACTIONS.index(last_action)

        reward = reward_from_events(events)

        old_q = self.q_table[state][action_index]

        # Terminal-State:
        # kein zukünftiger Q-Wert mehr
        new_q = old_q + ALPHA * (
            reward - old_q
        )

        self.q_table[state][action_index] = new_q

    # Q-Tabelle nach jeder Runde speichern
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.q_table, file)


def reward_from_events(events):
    """
    Reward-System für Task 1.

    Ziel:
    - Coins sammeln
    - in Richtung des nächsten Coins laufen
    - falsche Bewegungen bestrafen
    - ungültige Aktionen vermeiden
    """

    rewards = {
        # Hauptziel
        e.COIN_COLLECTED: 10,

        # Schlechte Aktionen
        e.INVALID_ACTION: -2,
        e.WAITED: -1,

        # Reward Shaping
        MOVED_TOWARDS_COIN: 0.5,
        NOT_TOWARDS_COIN: -0.2,
    }

    reward_sum = 0

    for event in events:
        if event in rewards:
            reward_sum += rewards[event]

    return reward_sum