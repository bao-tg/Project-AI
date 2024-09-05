"""
TODO: Implement Approximate Q-Learning player for Gomoku.
* Extract features from the state-action pair and store in a numpy array.
* Define the size of the feature vector in the feature_size method.
"""

from ..player import Player
from ..game import Gomoku

from copy import deepcopy
import random

class GMK_Reflex(Player):
    def __init__(self, letter):
        super().__init__(letter)
    
    def heuristic(self, board, player, row, col):
        modified_board = deepcopy(board)
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

    def no_threat(self, board_state, letter, size):
        for row in range(size):
            for col in range(size - 5 + 1):
                cnt = 0
                for i in range(5):
                    if board_state[row][col + i] != None:
                        cnt+= board_state[row][col + i] != letter
                if cnt == 4:
                    return False
                
        for col in range(size):
            for row in range(size - 5 + 1):
                cnt = 0
                for i in range(5):
                    if board_state[row + i][col] != None:
                        cnt+= board_state[row + i][col] != letter
                if cnt == 4:
                    return False
                 
        for row in range(size -  5 + 1):
            for col in range(size - 5 + 1):
                cnt = 0
                for i in range(5):
                    if board_state[row + i][col + i] != None:
                        cnt+= board_state[row + i][col + i] != letter
                    if cnt == 4:
                        return False
         
        for row in range(size - 5 + 1):
            for col in range(4, size):
                cnt = 0
                for i in range(5):
                    if board_state[row + i][col - i] != None:
                        cnt+= board_state[row + i][col - i] != letter
                if cnt == 4:
                    return False
                
        return True

    def policy_one(self, sequences):
        for seq in sequences:
            cnt = 0
            move = None
            for cell in seq:
                if cell[0] == None:
                    move = cell[1]
                elif cell[0] == self.letter:
                    cnt+= 1
            if cnt == 4:
                if move != None:
                    return move
                
        return None
    
    def policy_two(self, sequences):
        for seq in sequences:
            cnt = 0
            move = None
            for cell in seq:
                if cell[0] == None:
                    move = cell[1]
                elif cell[0] != self.letter:
                    cnt+= 1
            if cnt == 4:
                if move != None:
                    return move
        return None
        
    def policy_three(self, sequences, game):
        for seq in sequences:
            potential = True
            for cell in seq:
                if cell[0] != None and cell[0] != self.letter:
                    potential = False
                    break
            if potential == False:
                continue
            
            cnt = 0
            move = None
            for i in range(4):
                cell = seq[i]
                if cell[0] == None:
                    move = cell[1]
                    continue
                assert(cell[0] == self.letter)
                cnt+= 1
            if cnt == 3:
                cell = seq[0][1]
                type = seq[0][2]
                if 0 <= cell[0] - type[0] and 0 <= cell[1] - type[1]:
                    if game.board_state[cell[0] - type[0]][cell[1] - type[1]] == None or game.board_state[cell[0] - type[0]][cell[1] - type[1]] == self.letter:
                        return move

            cnt = 0
            move = None
            for i in range(1, 5):
                cell = seq[i]
                if cell[0] == None:
                    move = cell[1]
                    continue
                assert(cell[0] == self.letter)
                cnt+= 1
            if cnt == 3:
                cell = seq[-1][1]
                type = seq[-1][2]
                if cell[0] + type[0] < game.size and cell[1] + type[1] < game.size:
                    if game.board_state[cell[0] + type[0]][cell[1] + type[1]] == None or game.board_state[cell[0] + type[0]][cell[1] + type[1]] == self.letter:
                        return move
        return None
    
    def policy_four(self, sequences, game):
        freq = {}
        for row in range(game.size):
            for col in range(game.size):
                freq[(row, col)] = 0
        for seq in sequences:
            potential = True
            cnt = 0
            tmp_arr = []
            for cell in seq:
                if cell[0] == None:
                    tmp_arr.append(cell)
                    continue
                if cell[0] != self.letter:
                    potential = False
                    break
                cnt+= 1
            if potential == False or cnt != 3:
                continue
            for cell in tmp_arr:
                freq[cell[1]]+= 1
            
        for cell in freq.keys():
            if freq[cell] >= 2:
                return cell
        return None

    def policy_five(self, sequences, game):
        freq = {}
        for row in range(game.size):
            for col in range(game.size):
                freq[(row, col)] = False
        for seq in sequences:
            potential = True
            cnt = 0
            tmp_arr = []
            for cell in seq:
                if cell[0] != None:
                    tmp_arr.append(cell)
                    continue
                if cell[0] != self.letter:
                    potential = False
                    break
                cnt+= 1
            if potential == False or cnt != 3:
                continue
            

            assert(len(tmp_arr) == 2)
            for i in len(tmp_arr):
                cell = tmp_arr[i]
                freq[cell[1]] = True

        for seq in sequences:
            potential = True
            empty_cells = []
            cnt = 0
            for cell in seq:
                if cell[0] == None:
                    empty_cells.append(cell)
                    continue
                if cell[0] != self.letter:
                    potential = False
                    break
                cnt+= 1
            if potential == False or cnt != 2:
                continue
            
            if seq[0][0] != None and seq[-1][0] != None :
                continue
            
            potential = True
            cnt = 0
            for i in range(4):
                if seq[i][0] != None:
                    cnt+= 1
            cell = seq[0]
            type = cell[2]
            if cell[1][0] - type[0] < 0 or cell[1][1] - type[1] < 0 or cell[1][0] - type[0] >= game.size or cell[1][1] - type[1] >= game.size:
                potential = False
            else:
                x = cell[1][0] - type[0]
                y = cell[1][1] - type[1]
                if game.board_state[x][y] != None and game.board_state[x][y] != self.letter:
                    potential = False
            if cnt == 2 and potential == True:
                for i in range(4):
                    cell = seq[i]
                    if cell[0] != None:
                        continue
                    if freq[cell[1]] == True:
                        return cell[1]
            
            potential = True
            cnt = 0
            for i in range(1, 5):
                if seq[i][0] != None:
                    cnt+= 1
            cell = seq[4]
            type = cell[2]
            if cell[1][0] + type[0] < 0 or cell[1][1] + type[1] < 0 or cell[1][0] + type[0] >= game.size or cell[1][1] + type[1] >= game.size:
                potential = False
            elif game.board_state[cell[1][0] + type[0]][cell[1][1] + type[1]] != None and game.board_state[cell[1][0] + type[0]][cell[1][1] + type[1]] != self.letter:
                potential = False
            if cnt == 2 and potential == True:    
                for i in range(1, 5):
                    cell = seq[i]
                    if cell[0] != None:
                        continue
                    if freq[cell[1]] == True:
                        return cell[1]
            
            
        return None

    def policy_six(self, sequences, game):
        potential_moves = []
        for seq in sequences:
            potential = True
            cnt = 0
            tmp_arr = []
            for cell in seq:
                if cell[0] == None:
                    tmp_arr.append(cell)
                    continue
                if cell[0] != self.letter:
                    potential = False
                    break
                cnt+= 1
            if potential == False or cnt != 3:
                continue
            
            assert(len(tmp_arr) == 2)
            for i in range(len(tmp_arr)):
                cell = tmp_arr[i]
                potential_moves.append(cell[1])
        
        if len(potential_moves) > 0:
            best_cnt = 0
            best_cell = 0
            for cell in potential_moves:
                cnt = self.heuristic(game.board_state, self.letter, cell[0], cell[1])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_cell = cell
            return best_cell
        return None

    def get_move(self, game):
        sequences = game.get_sequences()
        
       
        move = self.policy_one(sequences) 
        if move != None:
            # print("Use policy one")
            return move
        
        move = self.policy_two(sequences)
        if move != None:
            # print("Use policy two")
            return move
        
        move = self.policy_three(sequences, game)
        if move != None:
            # print("Use policy three")
            return move
       
        move = self.policy_four(sequences, game)
        if move != None:
            # print("Use policy four")
            return move 
        
        move = self.policy_five(sequences, game)
        if move != None:
            # print("Use policy five")
            return move
        
        move = self.policy_six(sequences, game)
        if move != None:
            # print("Use policy six")
            return move
    
    def __str__(self):
        return "Reflex Player"