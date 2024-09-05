"""
TODO: Implement Approximate Q-Learning player for Gomoku.
* Extract features from the state-action pair and store in a dictionary with the format {feature_name: feature_value}.
* Similarly, the weight will be the dictionary in the format {feature_name: weight_value}.
* self.action_history stores the state-action pair for each move in the game.
* We don't use hashing for the board state as we need to extract features.
"""
import math
from typing import List, Tuple, Union, DefaultDict
from tqdm import tqdm
from ..player import Player
from collections import defaultdict
from .bots.intermediate import GMK_Intermediate

import numpy as np
import random
import os
import pickle

NUM_EPISODES = 200
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1
SEED = 2024
random.seed(SEED)

class GMK_ApproximateQPlayer(Player):
    def __init__(self, letter, size=15, transfer_player=GMK_Intermediate):
        super().__init__(letter)
        self.opponent = transfer_player
        self.num_episodes = NUM_EPISODES
        self.learning_rate = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EXPLORATION_RATE
        self.weights = defaultdict(float) # Initialize weights to 0
        self.action_history = []
        self.board_size = size
        self.feature_extractor = SimpleExtractor()
        self.weights['#-of-unblocked-five-player'] = 10000
        self.weights['#-of-unblocked-three-opponent'] = -30
        self.weights['#-of-unblocked-three-player'] = -30
        self.weights['#-of-unblocked-four-opponent'] = -130
        self.weights['#-of-unblocked-four-player'] = 130
        self.weights['#-of-unblocked-two-player'] = 7
        self.weights['#-of-unblocked-two-opponent'] = -7
        self.weights['bias'] = 1.0

    def train(self, game, save_filename=None):
        # Main Q-learning algorithm
        opponent_letter = 'X' if self.letter == 'O' else 'O'
        if self.opponent is None:
            opponent = GMK_ApproximateQPlayer(opponent_letter)
        else:
            opponent = self.opponent(opponent_letter)
            
        print(f"Training {self.letter} player for {self.num_episodes} episodes...")
        game_state = game.copy()
        
        for _ in tqdm(range(self.num_episodes)):               
            game_state.restart()
            opponent.action_history = []
            
            current_player = self if self.letter == 'X' else opponent 
            next_player = self if self.letter == 'O' else opponent
            while True:                
                if isinstance(current_player, GMK_ApproximateQPlayer):     
                    action = current_player.choose_action(game_state)
                    state = copy.deepcopy(game_state.board_state)
                    current_player.action_history.append((state, action)) 
                else:
                    action = current_player.get_move(game_state)
                
                next_game_state = game_state.copy()
                next_game_state.set_move(action[0], action[1], current_player.letter)
                
                if next_game_state.game_over():
                    reward = 1 if next_game_state.wins(current_player.letter) else -1 if next_game_state.wins(next_player.letter) else 0
                    if isinstance(current_player, GMK_ApproximateQPlayer):
                        current_player.update_rewards(reward)
                    if isinstance(next_player, GMK_ApproximateQPlayer):
                        next_player.update_rewards(-reward)
                    break
                else: 
                    current_player, next_player = next_player, current_player
                    game_state = next_game_state    

            self.letter = 'X' if self.letter == 'O' else 'O'
            opponent.letter = 'X' if opponent.letter == 'O' else 'O'  
            self.action_history = []
        
        print("Training complete. Saving training weights...")
        if save_filename is None:
            save_filename = f'{self.board_size}x{self.board_size}_{NUM_EPISODES}.pkl'
        self.save_weight(save_filename)
    
    def update_rewards(self, reward: float):
        """
        Given the reward at the end of the game, update the weights for each state-action pair in the game with the TD update rule:
            for weight w_i of feature f_i for (s, a):
                w_i = w_i + alpha * (reward + gamma * Q(s', a') - Q(s, a)) * f_i(s, a)

        * We need to update the Q-values for each state-action pair in the action history because the reward is only received at the end.
        * Make a call to update_q_values() for each state-action pair in the action history.
        """
        for t in range(len(self.action_history) - 1, -1, -1):
            state, action = self.action_history[t]
            if t == len(self.action_history) - 1:
                next_state = None
            else:
                next_state, _ = self.action_history[t + 1]
            self.update_q_values(state, action, next_state, reward)
            reward *= self.gamma

    def choose_action(self, game) -> Union[List[int], Tuple[int, int]]:
        """
        Choose action with ε-greedy strategy.
        If random number < ε, choose random action.
        Else choose action with the highest Q-value.
        :return: action
        """
        state = copy.deepcopy(game.board_state)
        # Exploration-exploitation trade-off
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(game.empty_cells())
        else:
            # Choose the action with the highest Q-value
            max_q_value = -math.inf
            best_action = None
            for action in game.empty_cells():
                q_value = self.q_value(state, action)
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
            action = best_action
        return action

    def update_q_values(self, state, action, next_state, reward):
        """
        Given (s, a, s', r), update the weights for the state-action pair (s, a) using the TD update rule:
            for weight w_i of feature f_i for (s, a):
                w_i = w_i + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a)) * f_i(s, a)
        :return: None
        """
        features = self.feature_vector(state, action)
        q_state_action = self.q_value(state, action)
        if next_state is not None:
            max_q_next = self.max_q_value(next_state)
        else:
            max_q_next = 0
        for feature_name, feature_value in features.items():
            self.weights[feature_name] += self.learning_rate * (reward + self.gamma * max_q_next - q_state_action) * feature_value

    def feature_vector(self, state, action) -> DefaultDict[str, float]:
        """
        Extract the feature vector for a given state-action pair.
        :return: feature vector
        """
        return self.feature_extractor.get_features(copy.deepcopy(state), action, self.letter)

    def max_q_value(self, state):
        max_q_value = -math.inf
        for action in self.empty_cells(state):
            q_value = self.q_value(state, action)
            if q_value > max_q_value:
                max_q_value = q_value
        return max_q_value

    def q_value(self, state, action) -> float:
        """
        Compute the Q-value for a given state-action pair as the dot product of the feature vector and the weight vector.
        :return: Q-value
        """
        q_value = 0
        features = self.feature_vector(state, action)
        for feature_name, feature_value in features.items():
            q_value += self.weights[feature_name] * feature_value
        return q_value
    
    def save_weight(self, filename):
        """
        Save the weights of the feature vector.
        """
        path = 'project/gomoku/q_weights'
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/{filename}', 'wb') as f:
            pickle.dump(dict(self.weights), f)

    def load_weight(self, filename):
        """
        Load the Q-table.
        """
        path = 'project/gomoku/q_weights'
        if not os.path.exists(f'{path}/{filename}'):
            raise FileNotFoundError(f"Weight file '{filename}' not found.")
        with open(f'{path}/{filename}', 'rb') as f:
            dict_weights = pickle.load(f)
            self.weights.update(dict_weights)

    def get_move(self, game):
        self.epsilon = 0  # No exploration
        return self.choose_action(game)
    
    def empty_cells(self, board: List[List[str]]) -> List[Tuple[int, int]]:
        """
        Return a list of empty cells in the board.
        """
        return [(x, y) for x in range(len(board)) for y in range(len(board[0])) if board[x][y] is None]

    def __str__(self):
        return "Approximate Q-Learning Player"

########################### Feature Extractor ###########################
from abc import ABC, abstractmethod
import copy

class FeatureExtractor(ABC):
    @abstractmethod
    def get_features(self, state: List[List[str]], move: Union[List[int], Tuple[int]], player: str) -> DefaultDict[str, float]:
        """
        :param state: current board state
        :param move: move taken by the player
        :param player: current player
        :return: a dictionary {feature_name: feature_value}
        """
        pass

class IdentityExtractor(FeatureExtractor):
    def get_features(self, state, move, player):
        """
        Return 1.0 for all state action pair.
        """
        feats = defaultdict(float)
        key = self.hash_board(state)
        feats[(key, tuple(move))] = 1.0
        return feats
    
    def hash_board(self, board):
        key = ''
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'X':
                    key += '1'
                elif board[i][j] == 'O':
                    key += '2'
                else:
                    key += '0'
        return key


class SimpleExtractor(FeatureExtractor):
    def get_features(self, state, move, player):
        opponent = 'X' if player == 'O' else 'O'

        x, y = move
        state = np.array(state)
        state[x][y] = player

        feats = defaultdict(float)
        feats['#-of-unblocked-three-player'] = self.count_open_three(player, state)
        feats['#-of-unblocked-three-opponent']=self.count_open_three(opponent, state)
        feats['#-of-unblocked-four-player'] = self.count_open_four(player, state) 
        feats['#-of-unblocked-four-opponent'] = self.count_open_four(opponent, state)
        feats['#-of-unblocked-five-player'] = self.count_open_five(player, state)
        feats['#-of-unblocked-two-player'] = self.count_open_two(player, state)
        feats['#-of-unblocked-two-opponent'] = self.count_open_two(opponent, state)
        feats['bias'] = 1.0

        return feats


    def count_open_two(self, player, board):
        length = 5
        def check_open_four(player, array):
            lst = list(array)
            return lst.count(player) == 2 and lst.count(None) == 3
        threat_cnt = 0
        size = len(board)
        for row in range(size):
            for col in range(size - (length - 1)):
                array = board[row, col:col + length]
                if check_open_four(player, array):
                    threat_cnt += 1

        for col in range(size):
            for row in range(size - (length - 1)):
                array = board[row:row + length, col]
                if check_open_four(player, array):
                    threat_cnt += 1

        for row in range(size - (length - 1)):
            for col in range(size - (length - 1)):
                array = [board[row + i, col + i] for i in range(length)]
                if check_open_four(player, array):
                    threat_cnt += 1

                array = [board[row + i, col + length - 1 - i] for i in range(length)]
                if check_open_four(player, array):
                    threat_cnt += 1

        return threat_cnt

    def count_open_three(self, player, board):
        length = 5
        def check_open_three(player, array):
            lst = list(array)
            return lst.count(player) == 3 and lst.count(None) == 2
        
        threat_cnt = 0
        size = len(board)
        for row in range(size):
            for col in range(size - (length - 1)):
                array = board[row, col:col + length]
                if check_open_three(player, array):
                    threat_cnt += 1

        for col in range(size):
            for row in range(size - (length - 1)):
                array = board[row:row + length, col]
                if check_open_three(player, array):
                    threat_cnt += 1

        for row in range(size - (length - 1)):
            for col in range(size - (length - 1)):
                array = [board[row + i, col + i] for i in range(length)]
                if check_open_three(player, array):
                    threat_cnt += 1

                array = [board[row + i, col + length - 1 - i] for i in range(length)]
                if check_open_three(player, array):
                    threat_cnt += 1

        return threat_cnt
    

    def count_open_four(self, player, board):
        length = 5
        def check_open_four(player, array):
            lst = list(array)
            return lst.count(player) == 4 and lst.count(None) == 1
        threat_cnt = 0
        size = len(board)
        for row in range(size):
            for col in range(size - (length - 1)):
                array = board[row, col:col + length]
                if check_open_four(player, array):
                    threat_cnt += 1

        for col in range(size):
            for row in range(size - (length - 1)):
                array = board[row:row + length, col]
                if check_open_four(player, array):
                    threat_cnt += 1

        for row in range(size - (length - 1)):
            for col in range(size - (length - 1)):
                array = [board[row + i, col + i] for i in range(length)]
                if check_open_four(player, array):
                    threat_cnt += 1

                array = [board[row + i, col + length - 1 - i] for i in range(length)]
                if check_open_four(player, array):
                    threat_cnt += 1

        return threat_cnt

    def count_open_five(self, player, board):
        length = 5
        def check_open_four(player, array):
            lst = list(array)
            return lst.count(player) == 5 
        
        threat_cnt = 0
        size = len(board)
        for row in range(size):
            for col in range(size - (length - 1)):
                array = board[row, col:col + length]
                if check_open_four(player, array):
                    threat_cnt += 1

        for col in range(size):
            for row in range(size - (length - 1)):
                array = board[row:row + length, col]
                if check_open_four(player, array):
                    threat_cnt += 1

        for row in range(size - (length - 1)):
            for col in range(size - (length - 1)):
                array = [board[row + i, col + i] for i in range(length)]
                if check_open_four(player, array):
                    threat_cnt += 1

                array = [board[row + i, col + length - 1 - i] for i in range(length)]
                if check_open_four(player, array):
                    threat_cnt += 1

        return threat_cnt

    