from exceptions import GameplayException
from connect4 import Connect4
from randomagent import RandomAgent
from minmaxagent import MinMaxAgent
from alphabetaagent import AlphaBetaAgent

connect4 = Connect4(width=7, height=6)
agent = AlphaBetaAgent(connect4,'x',7,True)
while not connect4.game_over:
    connect4.draw()
    try:
        if connect4.who_moves == agent.token:
            n_column = agent.decide()
        else:
            n_column = int(input(':'))
        connect4.drop_token(n_column)
    except (ValueError, GameplayException):
        print('>> Invalid move!')

connect4.draw()
