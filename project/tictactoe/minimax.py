from typing import List, Tuple, Union
import random
import math
from ..player import Player
from ..game import TicTacToe

class TTT_MinimaxPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game: TicTacToe) -> Union[List[int], Tuple[int, int]]:
        depth = len(game.empty_cells())
        if depth == 9:  # First move of the game
            move = random.choice(game.empty_cells())
        else:
            _, move = self.minimax(game, depth, self.letter)
        return move

    def minimax(self, game: TicTacToe, depth: int, player_letter: str) -> Tuple[int, Tuple[int, int]]:
        if player_letter == self.letter:
            best = [-math.inf, None]
        else:
            best = [math.inf, None]
        
        if game.game_over() or depth == 0:
            score = self.evaluate(game)
            return [score, None]

        for cell in game.empty_cells():
            x, y = cell
            game.set_move(x, y, player_letter)  # Make the move
            score = self.minimax(game, depth - 1, 'O' if player_letter == 'X' else 'X')
            game.reset_move(x, y)  # Undo the move

            score[1] = (x, y)
            
            if player_letter == self.letter:
                if score[0] > best[0]:  # Maximize self
                    best = score
            else:
                if score[0] < best[0]:  # Minimize opponent
                    best = score

        return best

    def evaluate(self, game: TicTacToe) -> int:
        if game.wins(self.letter):
            return math.inf  # Favorable to self
        elif game.wins('O' if self.letter == 'X' else 'X'):
            return -math.inf  # Favorable to opponent
        return 0  # Neutral or draw

    def __str__(self) -> str:
        return "Minimax Player"
