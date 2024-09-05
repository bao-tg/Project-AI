"""
TODO: Implement a Minimax player with Alpha-Beta Pruning for Gomoku.
* You need to implement a heuristic evaluation function for non-terminal states.
* Optional: Implement the promising_next_moves function to reduce the branching factor.
"""
import copy
from ..player import Player
from ..game import Gomoku
from typing import List, Tuple, Union
import math
import random
import numpy as np

EMPTY = None
SEED = 2024
random.seed(SEED)
DEPTH = 2  # Define the depth of the search tree.

class GMK_AlphaBetaPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
        self.depth = DEPTH 
    
    def get_move(self, game: Gomoku):
        if game.last_move == (-1, -1):
            mid_size = game.size // 2
            moves = [(mid_size, mid_size), (mid_size - 1, mid_size - 1), (mid_size + 1, mid_size + 1), (mid_size - 1, mid_size + 1), (mid_size + 1, mid_size - 1)]
            move = random.choice(moves)
            while not game.valid_move(move[0], move[1]):
                move = random.choice(moves)
            return move 
        else:
            # Alpha-Beta Pruning: Initialize alpha to negative infinity and beta to positive infinity
            alpha = -math.inf
            beta = math.inf
            choice = self.minimax(game, self.depth, self.letter, alpha, beta)
            move = [choice[0], choice[1]]
        return move

    def minimax(self, game, depth, player_letter, alpha, beta) -> Union[List[int], Tuple[int]]:
        """
        AI function that chooses the best move with alpha-beta pruning.
        :param game: current state of the board
        :param depth: node index in the tree (0 <= depth <= 9)
        :param player_letter: value representing the player
        :param alpha: best value that the maximizer can guarantee
        :param beta: best value that the minimizer can guarantee
        :return: a list or a tuple with [best row, best col, best score]
        """
        if player_letter == self.letter:
            best = [-1, -1, -math.inf]  # Max Player
        else:
            best = [-1, -1, +math.inf]  # Min Player

        if depth == 0 or game.game_over():
            score = self.evaluate(game)
            return [-1, -1, score]

        for cell in self.promising_next_moves(game, player_letter):
            x, y = cell[0], cell[1]
            game.board_state[x][y] = player_letter
            other_letter = 'X' if player_letter == 'O' else 'O'
            game.last_move = (x, y)
            score = self.minimax(game, depth - 1, other_letter, alpha, beta)
            game.board_state[x][y] = None
            score[0], score[1] = x, y

            if player_letter == self.letter:  # Max player
                if score[2] > best[2]:
                    best = score
                alpha = max(alpha, best[2])
                if beta <= alpha:
                    break
            else:  # Min player
                if score[2] < best[2]:
                    best = score
                beta = min(beta, best[2])
                if beta <= alpha:
                    break
        return best

    def evaluate(self, game, state=None) -> float:
        """
        Heuristic evaluation function for the given state when reaching a leaf node.
        :return: a float value representing the score of the state
        """
        player_letter = self.letter
        opponent = 'X' if self.letter == 'O' else 'O'

        if game.wins(self.letter, state):
            return 10000000 * (len(game.empty_cells()) + 1)
        if game.wins(opponent, state):
            return -10000000 * (len(game.empty_cells()) + 1)

        straight_four_weight = 100000
        threat_weight = 10000000
        alignment_weight = 50000
        open_four_weight = 100000
        center_control_weight = 20000
        player_score = 0
        opponent_score = 0

        # Center position
        center_x, center_y = game.size // 2, game.size // 2

        # Evaluate threats and opportunities
        player_threats = self.find_threats(5, player_letter, np.array(game.board_state))
        opponent_threats = self.find_threats(5, opponent, np.array(game.board_state))
        player_straight_fours = self.find_threats(6, player_letter, np.array(game.board_state))
        opponent_straight_fours = self.find_threats(6, opponent, np.array(game.board_state))
        player_open_fours = self.find_threats(7, player_letter, np.array(game.board_state))
        opponent_open_fours = self.find_threats(7, opponent, np.array(game.board_state))

        player_score += player_threats * threat_weight
        opponent_score += opponent_threats * threat_weight
        player_score += player_straight_fours * straight_four_weight
        opponent_score += opponent_straight_fours * straight_four_weight
        player_score += player_open_fours * open_four_weight
        opponent_score += opponent_open_fours * open_four_weight

        # Additional heuristic: count max aligned pieces for both players and control center
        for row in range(game.size):
            for col in range(game.size):
                if game.board_state[row][col] == player_letter:
                    player_score += self.heuristic(game.board_state, player_letter, row, col) * alignment_weight
                    # Control center score
                    player_score += center_control_weight / (1 + abs(center_x - row) + abs(center_y - col))
                elif game.board_state[row][col] == opponent:
                    opponent_score += self.heuristic(game.board_state, opponent, row, col) * alignment_weight
                    # Control center score
                    opponent_score += center_control_weight / (1 + abs(center_x - row) + abs(center_y - col))

        return player_score - opponent_score
    
    def promising_next_moves(self, game, player_letter) -> List[Tuple[int]]:
        """
        Identify promising next moves to reduce the search space.
        :return: a list of tuples with the best potential moves
        """
        moves = []
        min_x, max_x, min_y, max_y = game.size, -1, game.size, -1

        # Determine the bounds of the existing pieces on the board
        for x in range(game.size):
            for y in range(game.size):
                if game.board_state[x][y] is not None:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)

        # Expand the bounds slightly to consider neighboring positions
        min_x = max(0, min_x - 2)
        max_x = min(game.size - 1, max_x + 2)
        min_y = max(0, min_y - 2)
        max_y = min(game.size - 1, max_y + 2)

        # Collect all empty positions within the expanded bounds
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if game.board_state[x][y] is None:
                    moves.append((x, y))

        return moves

    def find_threats(self, length, player, board):
        threat_cnt = 0
        size = len(board)
        for row in range(size):
            for col in range(size-(length-1)):
                array = board[row,col:col+length]
                is_threat = self.get_threat(player, array)
                if is_threat: 
                    threat_cnt += 1              
                    
        ## Read vertically
        for col in range(size):
            for row in range(size-(length-1)):
                array = board[row:row+length,col]
                is_threat = self.get_threat(player, array)
                if is_threat: 
                    threat_cnt += 1

        ## Read diagonally
        for row in range(size-(length-1)):
            for col in range(size-(length-1)):
                array = []
                for i in range(length):
                    array.append(board[i+row,i+col])
                is_threat = self.get_threat(player, array)
                if is_threat: 
                    threat_cnt += 1              

                array = []
                for i in range(length):
                    array.append(board[i+row,col+length-1-i])
                is_threat = self.get_threat(player, array)
                if is_threat: 
                    threat_cnt += 1 
        return threat_cnt
        
    def get_threat(self, player, array):
        if len(array) == 5:
            x = list(array)
            if x.count(player) == 4 and x.count(EMPTY) == 1:
                return True
            return False
        # Threat:  
        elif len(array) == 6:
            if array[0] == EMPTY and array[1] == player and array[5] == EMPTY and array[4] == player:
                if (array[2] == EMPTY and array[3] == player):
                    return True
                elif (array[2] == player and array[3] == EMPTY):
                    return True
            return False
        
        elif len(array) == 7:
            opp = 'X' if player == 'O' else 'O'
            if array[1] == EMPTY and array[2] == player and array[3] == player and array[4] == player and array[5] == EMPTY:
                if array[0] == EMPTY and array[6] == EMPTY:
                    return True
                elif array[0] == opp and array[6] == EMPTY:
                    return True
                elif array[0] == EMPTY and array[6] == opp:
                    return True
            return False
    
    def is_winning_move(self, game, x, y, player):
        modified_game = game.copy()
        modified_game.set_move(x, y, player)
        return modified_game.wins(player)
    
    def is_straight_four(self, game, x, y, player):
        board = np.array(copy.deepcopy(game.board_state))
        board[x][y] = player
        size = len(board)
        length = 6

        # Read horizontally
        for row in range(size):
            for col in range(size-(length-1)):
                array = board[row,col:col+length]
                is_threat = self.straight_four(player, array)
                if is_threat: 
                    return True            
                    
        ## Read vertically
        for col in range(size):
            for row in range(size-(length-1)):
                array = board[row:row+length,col]
                is_threat = self.straight_four(player, array)
                if is_threat: 
                    return True

        ## Read diagonally
        for row in range(size-(length-1)):
            for col in range(size-(length-1)):
                array = []
                for i in range(length):
                    array.append(board[i+row,i+col])
                is_threat = self.straight_four(player, array)
                if is_threat: 
                    return True             

                array = []
                for i in range(length):
                    array.append(board[i+row,col+length-1-i])
                is_threat = self.straight_four(player, array)
                if is_threat: 
                    return True
        return False
    
    def heuristic(self, board, player, row, col):
        modified_board = copy.deepcopy(board)
        modified_board[row][col] = player
        board_size = len(board)

        max_count = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < board_size and 0 <= c < board_size and modified_board[r][c] == player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < board_size and 0 <= c < board_size and modified_board[r][c] == player:
                    count += 1
                else:
                    break
            max_count += count
        return max_count
    
    def __str__(self):
        return "AlphaBeta Player"
