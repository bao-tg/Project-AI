"""
TODO: Implement the TTT_QPlayer class.
* Note 1: You should read the game logic in project/game.py to familiarize yourself with the environment.
* Note 2: You don't have to strictly follow the template or even use it at all. Feel free to create your own implementation.
"""

import random
from typing import List, Tuple, Union
from ..player import *
from ..game import TicTacToe
from . import *
from tqdm import tqdm

NUM_EPISODES = 1000000
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1


class TTT_QPlayer(RandomPlayer):
    def __init__(self, letter, transfer_player=None):
        super().__init__(letter)
        self.opponent = transfer_player
        self.num_episodes = NUM_EPISODES
        self.learning_rate = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EXPLORATION_RATE
        self.Q = {}
        self.action_history = []

    def train(self, game):
        """
        Train the Q-Learning player against an transfer player to update the Q tables.
        """
        opponent_letter = "X" if self.letter == "O" else "O"
        if self.opponent is None:
            opponent = TTT_QPlayer(opponent_letter)
        else:
            opponent = self.opponent(opponent_letter)

        print(f"Training Q Player [{self.letter}] for {self.num_episodes} episodes...")
        game_state = game.copy()

        for _ in tqdm(range(self.num_episodes)):
            game_state.restart()
            self.action_history = []
            opponent.action_history = []

            current_player = self if self.letter == "X" else opponent
            next_player = self if self.letter == "O" else opponent

            while True:
                if isinstance(current_player, TTT_QPlayer):
                    action = current_player.choose_action(game_state)
                    state = current_player.hash_board(game_state.board_state)
                    current_player.action_history.append((state, action))
                else:
                    action = current_player.get_move(game_state)

                next_game_state = game_state.copy()
                next_game_state.set_move(action[0], action[1], current_player.letter)

                if next_game_state.game_over():
                    reward = (
                        1
                        if next_game_state.wins(current_player.letter)
                        else -1
                        if next_game_state.wins(next_player.letter)
                        else 0
                    )
                    if isinstance(current_player, TTT_QPlayer):
                        current_player.update_rewards(reward)
                    if isinstance(next_player, TTT_QPlayer):
                        next_player.update_rewards(-reward)
                    break
                else:
                    current_player, next_player = next_player, current_player
                    game_state = next_game_state

            self.letter = "X" if self.letter == "O" else "O"
            opponent.letter = "X" if opponent.letter == "O" else "O"

    def update_rewards(self, reward: float):
        """
        Adjust the Q-values for each state-action pair at the end of the game using the Bellman equation:
            Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
        Iterate backwards through the action history, updating Q-values from the end of the game toward the beginning.
        """
        for i in reversed(range(len(self.action_history))):
            state, action = self.action_history[i]
            action = tuple(action)
            if state not in self.Q:
                self.Q[state] = {}
            if action not in self.Q[state]:
                self.Q[state][action] = 0.0

            # Calculate the maximum future Q-value
            future_q = 0
            if i + 1 < len(self.action_history):
                next_state, next_action = self.action_history[i + 1]
                if next_state in self.Q:
                    future_q = max(self.Q[next_state].values(), default=0)

            # Update Q-value using the Bellman equation
            self.Q[state][action] += self.learning_rate * (reward + self.gamma * future_q - self.Q[state][action])

            # Only the immediate reward is updated, future rewards are considered via future Q-values
            reward = 0  # Reset reward for the next iteration

    def choose_action(self, game: TicTacToe) -> Union[List[int], Tuple[int, int]]:
        """
        Apply ε-greedy strategy to select an action:
        With probability ε, take a random action among available moves.
        Otherwise, act optimally by choosing the action with the highest Q-value in the current state.
        """
        state = self.hash_board(game.board_state)
        if random.random() < self.epsilon:
            # Random action based on exploration rate
            return random.choice(game.empty_cells())
        if state in self.Q and self.Q[state]:
            # Optimal action based on the highest Q-value
            return max(self.Q[state], key=self.Q[state].get)
        # Fallback to a random action if no Q-value exists
        return random.choice(game.empty_cells())


    def hash_board(self, board):
        key = ""
        for i in range(3):
            for j in range(3):
                if board[i][j] == "X":
                    key += "1"
                elif board[i][j] == "O":
                    key += "2"
                else:
                    key += "0"
        return key

    def get_move(self, game: TicTacToe):
        self.epsilon = 0  # No exploration
        move = self.choose_action(game)
        return move

    def __str__(self):
        return "Q-Learning Player"
