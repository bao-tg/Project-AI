"""
TODO: Implement the standard MCTS player for Gomoku.
* tree policy: UCB1
* rollout policy: random
"""

from copy import deepcopy
import numpy as np
import math

from project.gomoku.alphabeta import GMK_AlphaBetaPlayer
from project.gomoku.reflex import GMK_Reflex

from ..player import Player
from ..game import Gomoku

WIN = 1
LOSE = -1
DRAW = 0
NUM_SIMULATIONS = 1000000
DEPTH = 10

import random
SEED = 2024
random.seed(SEED)

class TreeNode():
    def __init__(self, game_state: Gomoku, player_letter: str, parent=None, parent_action=None):
        self.player = player_letter
        self.game_state = game_state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.N = 0
        self.Q = 0
        self.ab_depth = DEPTH
    
    def select(self) -> 'TreeNode':
        """
        Select the best child node based on UCB1 formula. Keep selecting until a leaf node is reached.
        """
        current_node = self
        while not current_node.is_leaf_node():
            current_node = current_node.best_child()
        return current_node
    
    def expand(self) -> 'TreeNode':
        """
        Expand the current node by adding all possible child nodes. Return one of the child nodes for simulation.
        """
        minimax_agent = GMK_AlphaBetaPlayer(self.player)
        possible_moves = minimax_agent.promising_next_moves(self.game_state, self.player)
        for move in possible_moves:
            child_game_state = self.game_state.copy()
            child_game_state.set_move(move[0], move[1], self.player)
            child_player = 'X' if self.player == 'O' else 'O'
            child_node = TreeNode(child_game_state, child_player, parent=self, parent_action=move)
            self.children.append(child_node)
    
    def simulate(self) -> int:
        """
        Run simulation from the current node until the game is over. Return the result of the simulation.
        """
        player_letter = self.player
        opponent_letter = 'X' if player_letter == 'O' else 'O'
        
        curr_letter = player_letter
        simulate_game = self.game_state.copy()
        depth = self.ab_depth

        while True:
            if simulate_game.wins(player_letter):
                return WIN
            elif simulate_game.wins(opponent_letter):
                return LOSE
            elif len(simulate_game.empty_cells()) == 0:
                return DRAW
            elif depth == 0:
                minimax_agent = GMK_AlphaBetaPlayer(curr_letter)
                score = minimax_agent.evaluate(deepcopy(simulate_game))
                # CHECK WHETHER THE AGENT IS LIKELY TO WIN OR NOT
                if score > 0:
                    if curr_letter == player_letter:
                        return WIN
                    else:
                        return LOSE
                elif score < 0:
                    if curr_letter == player_letter:
                        return LOSE
                    return WIN
                return DRAW
            else:   
                reflex_agent = GMK_Reflex(curr_letter)
                move = reflex_agent.get_move(simulate_game)
                simulate_game.set_move(move[0], move[1], curr_letter)
                curr_letter = 'X' if curr_letter == 'O' else 'O'
                depth -= 1
    
    def backpropagate(self, result: int):
        if self.parent:
            self.parent.backpropagate(-result)
        self.N += 1
        self.Q += result
            
    def is_leaf_node(self) -> bool:
        return len(self.children) == 0
    
    def is_terminal_node(self) -> bool:
        return self.game_state.game_over()
    
    def best_child(self) -> 'TreeNode':
        return max(self.children, key=lambda c: c.ucb())
    
    def ucb(self, c=math.sqrt(2)) -> float:
        return self.Q / (1+self.N) + c * np.sqrt(np.log(self.parent.N) / (1+self.N))
    
class GMK_BetterMCTS(Player):
    def __init__(self, letter, num_simulations=NUM_SIMULATIONS):
        super().__init__(letter)
        self.num_simulations = num_simulations
    
    def get_move(self, game: Gomoku):
        mtcs = TreeNode(game, self.letter)
        for num in range(self.num_simulations):
            leaf = mtcs.select()
            if not leaf.is_terminal_node():
                leaf.expand()
            result = leaf.simulate()
            leaf.backpropagate(-result)
            
        best_child = max(mtcs.children, key=lambda c: c.N)
        return best_child.parent_action
    
    def __str__(self) -> str:
        return "Better MCTS Player"