# Importowanie wymaganych modułów i klas
from exceptions import GameplayException
from exceptions import AgentException
import copy


class MinMaxAgent:
    def __init__(self, game, token, depth, evaluation):
        self.game = game
        self.token = token
        self.opponent_token = 'x' if token == 'o' else 'o'
        self.depth = depth
        self.evaluation = evaluation

    def decide(self):
        _, move = self.minimax(self.game, self.depth, True)
        if move is None:
            raise AgentException('>> No valid moves!')
        return move

    def minimax(self, game, depth, maximizing_player):
        if game.game_over:
            if game.game_over and game.wins == self.token:
                return 1, None
            elif game.game_over and game.wins == self.opponent_token:
                return -1, None
            else:
                return 0, None
        # Jeśli osiągnięto maksymalną głębokość przeszukiwania, zwróć heurystyczną ocenę stanu gry
        elif depth == 0:
            return (self.evaluate(game), None) if self.evaluation else (0, None)

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in game.possible_drops():
                game_copy = game.copy()
                game_copy.drop_token(move)
                # Rekurencyjne wywołanie metody minimax dla gry po wykonaniu ruchu
                eval, _ = self.minimax(game_copy, depth - 1, False)
                # Jeśli ocena jest większa od aktualnej maksymalnej oceny, zaktualizuj maksymalną ocenę i najlepszy ruch
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in game.possible_drops():
                game_copy = game.copy()
                game_copy.drop_token(move)
                # Rekurencyjne wywołanie metody minimax dla gry po wykonaniu ruchu
                eval, _ = self.minimax(game_copy, depth - 1, True)
                # Jeśli ocena jest mniejsza od aktualnej minimalnej oceny, zaktualizuj minimalną ocenę i najlepszy ruch
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            return min_eval, best_move

    def evaluate(self, game):
        # Zliczanie sąsiadujących pól dla naszego agenta i przeciwnika
        my_count = self.count_neighborhood(game, self.token)
        enemy_count = self.count_neighborhood(game, self.opponent_token)
        # Różnica między liczbą sąsiadujących pól naszego agenta a przeciwnika, podzieloną przez maksymalną liczbę sąsiadujących pól plus jeden
        return (my_count - enemy_count) / (max(my_count, enemy_count) + 1)

    def count_neighborhood(self, game, token):
        count = 0
        for row in range(game.height):
            for col in range(game.width):
                # Jeśli pole należy do naszego agenta, zlicz sąsiadujące pola
                if game.board[row][col] == self.token:
                    neighbors_count = self.count_neighbors(row, col, token)
                    count += max(0, neighbors_count - 1)
        return count

    def count_neighbors(self, row, col, token):
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1)]
        neighbor_count = 0
        for drow, dcol in directions:
            r, c = row + drow, col + dcol
            # Zliczanie sąsiadów w danym kierunku dla danego pola
            while 0 <= r < self.game.height and 0 <= c < self.game.width and self.game.board[r][c] == token:
                neighbor_count += 1
                r += drow
                c += dcol
        return neighbor_count