import numpy as np
import random
import math

from ..player import Player
from ..game import TicTacToe

WIN = 1
LOSE = -1
DRAW = 0
NUM_SIMULATIONS = 50

class TreeNode():
    def __init__(self, game_state: TicTacToe, player_letter: str, parent=None, parent_action=None):
        self.player = player_letter
        self.game_state = game_state.copy()  # Create a copy of the game state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.N = 0  # Number of visits
        self.Q = 0  # Total value

    def select(self) -> 'TreeNode':
        # Selecting the child with the highest UCB value
        current = self
        while not current.is_leaf_node():
            if current.is_terminal_node():
                return current
            current = max(current.children, key=lambda node: node.ucb())
        return current

    def expand(self):
        # Expanding the node by creating new child nodes for all valid moves
        moves = self.game_state.empty_cells()
        for move in moves:
            new_state = self.game_state.copy()
            new_state.set_move(move[0], move[1], self.player)
            next_player = 'O' if self.player == 'X' else 'X'
            child_node = TreeNode(new_state, next_player, self, move)
            self.children.append(child_node)
        if len(self.children) > 0:
            return random.choice(self.children)
        return None

    def simulate(self):
        # Randomly playing moves until the game ends
        current_state = self.game_state.copy()
        current_player = self.player
        while not current_state.game_over():
            possible_moves = current_state.empty_cells()
            if not possible_moves:
                break
            move = random.choice(possible_moves)
            current_state.set_move(move[0], move[1], current_player)
            current_player = 'O' if current_player == 'X' else 'X'
        if current_state.wins(self.player):
            return WIN
        elif current_state.wins('O' if self.player == 'X' else 'X'):
            return LOSE
        return DRAW

    def backpropagate(self, result):
        # Backpropagation step, updating node statistics
        node = self
        while node is not None:
            node.N += 1
            node.Q += result
            result = -result  # Reverse the result for the opponent
            node = node.parent

    def is_leaf_node(self):
        return len(self.children) == 0

    def is_terminal_node(self):
        return self.game_state.game_over()

    def ucb(self, c=2):
        if self.N == 0:
            return float('inf')
        return self.Q / self.N + c * np.sqrt(np.log(self.parent.N) / self.N)
    
class TTT_MCTSPlayer(Player):
    def __init__(self, letter, num_simulations=NUM_SIMULATIONS):
        super().__init__(letter)
        self.num_simulations = num_simulations
    
    def get_move(self, game):
        root = TreeNode(game, self.letter)
        for _ in range(self.num_simulations):
            leaf = root.select()
            if not leaf.is_terminal_node():
                child = leaf.expand()
                if child is not None:
                    leaf = child
            result = leaf.simulate()
            leaf.backpropagate(-result)
        
        best_child = max(root.children, key=lambda c: c.N)
        return best_child.parent_action
    
    def __str__(self) -> str:
        return "MCTS Player"
