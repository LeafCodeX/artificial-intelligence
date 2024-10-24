import random

from exceptions import AgentException


class RandomAgent:
    def __init__(self, game,token='o'):
        self.token = token
        self.game = game

    def decide(self):
        if self.game.who_moves != self.token:
            raise AgentException('>> Not my round!')
        return random.choice(self.game.possible_drops())
