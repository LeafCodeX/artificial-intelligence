import time
from collections import defaultdict
from exceptions import GameplayException
from connect4 import Connect4
from randomagent import RandomAgent
from minmaxagent import MinMaxAgent
from alphabetaagent import AlphaBetaAgent

num_games = 3

results = defaultdict(int)
times = defaultdict(float)

for game in range(num_games):
    connect4 = Connect4(width=7, height=6)
    #agent1 = RandomAgent(connect4,'o')
    #agent2 = AlphaBetaAgent(connect4,'x',5,True)
    #agent1 = RandomAgent(connect4,'o')
    #agent2 = MinMaxAgent(connect4,'x',5,True)
    #agent1 = AlphaBetaAgent(connect4,'o',4,True)
    #agent2 = MinMaxAgent(connect4,'x',4,True)
    agent1 = MinMaxAgent(connect4,'o',4,False)
    agent2 = MinMaxAgent(connect4,'x',4,True)
    num_moves = 0

    while not connect4.game_over:
        try:
            start_time = time.time()
            if connect4.who_moves == agent1.token:
                n_column = agent1.decide()
            else:
                n_column = agent2.decide()
            times[connect4.who_moves] += time.time() - start_time
            num_moves += 1
            connect4.drop_token(n_column)
            if game == 0:
                print(f"==========================================")
                print(f">> Move {num_moves} in Game {game+1}:")
                connect4.draw()
        except (ValueError, GameplayException):
            print('>> Invalid move!')

    results[connect4.wins if connect4.wins else 'draw'] += 1
    print(f"Game {game+1}/{num_games}! Win => [{connect4.wins if connect4.wins else 'draw'}] with times: ({agent1.__class__.__name__}): {times['o']/num_moves:.4f}s vs ({agent2.__class__.__name__}): {times['x']/num_moves:.4f}s")

print(f"Results of {num_games} games: {dict(results)}")
print(f"Average time move: ({agent1.__class__.__name__}): {times['o']/num_moves:.4f}s vs ({agent2.__class__.__name__}): {times['x']/num_moves:.4f}s")
print(f"===========================================================================================")