from exceptions import AgentException

class AlphaBetaAgent:
    def __init__(self, game, token, depth, evaluation):
        self.game = game
        self.token = token
        self.opponent_token = 'x' if token == 'o' else 'o'
        self.depth = depth
        self.evaluation = evaluation

    def decide(self):
        _, move = self.alphabeta(self.game, self.depth, True, float('-inf'), float('inf'))
        if move is None:
            raise AgentException('>> No valid moves!')
        return move

    def alphabeta(self, game, depth, maximizing_player, alpha, beta):
        if game.game_over:
            if game.wins == self.token:
                return 1, None
            elif game.wins == self.opponent_token:
                return -1, None
            else:
                return 0, None
        elif depth == 0:
            return (self.evaluate(game), None) if self.evaluation else (0, None)

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in game.possible_drops():
                game_copy = game.copy()
                game_copy.drop_token(move)
                eval, _ = self.alphabeta(game_copy, depth - 1, False, alpha, beta)
                alpha = max(alpha, eval)
                # if ocena > od aktualnej maksymalnej oceny, zaktualizuj maksymalną ocenę i najlepszy ruch
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                # Pruning
                if alpha >= beta:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in game.possible_drops():
                game_copy = game.copy()
                game_copy.drop_token(move)
                eval, _ = self.alphabeta(game_copy, depth - 1, True, alpha, beta)
                beta = min(beta, eval)
                # if ocena < od aktualnej minimalnej oceny, zaktualizuj minimalną ocenę i najlepszy ruch
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                # Pruning
                if alpha >= beta:
                    break
            return min_eval, best_move

    def evaluate(self, game):
        # Zliczanie sąsiadujących pól dla naszego agenta i przeciwnika
        my_count = self.count_neighborhood(game, self.token)
        enemy_count = self.count_neighborhood(game, self.opponent_token)
        # Różnica między liczbą sąsiadujących pól agenta, a przeciwnika podzieloną przez maksymalną liczbę sąsiadujących pól plus jeden
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