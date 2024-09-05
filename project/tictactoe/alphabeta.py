import random
import math
from ..player import Player
from ..game import TicTacToe

class TTT_AlphaBetaPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game: TicTacToe):
        depth = len(game.empty_cells())
        if depth == 0 or game.game_over():
            return None

        if depth == 9:  # First move of the game
            move = random.choice(game.empty_cells())
        else:
            _, move = self.minimax(game, depth, self.letter, -math.inf, math.inf)
        return move

    def minimax(self, game: TicTacToe, depth: int, player_letter: str, alpha: float, beta: float) -> tuple:
        if game.game_over() or depth == 0:
            score = self.evaluate(game)
            return score, None

        if player_letter == self.letter:
            best_value = -math.inf
            best_move = None
            for cell in game.empty_cells():
                x, y = cell
                game.set_move(x, y, player_letter)
                value, _ = self.minimax(game, depth - 1, 'O' if player_letter == 'X' else 'X', alpha, beta)
                game.reset_move(x, y)
                if value > best_value:
                    best_value = value
                    best_move = (x, y)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return best_value, best_move
        else:
            best_value = math.inf
            best_move = None
            for cell in game.empty_cells():
                x, y = cell
                game.set_move(x, y, player_letter)
                value, _ = self.minimax(game, depth - 1, 'X' if player_letter == 'O' else 'O', alpha, beta)
                game.reset_move(x, y)
                if value < best_value:
                    best_value = value
                    best_move = (x, y)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return best_value, best_move

    def evaluate(self, game: TicTacToe) -> int:
        if game.wins(self.letter):
            return math.inf
        elif game.wins('O' if self.letter == 'X' else 'X'):
            return -math.inf
        return 0

    def __str__(self) -> str:
        return "Alpha-Beta Player"
