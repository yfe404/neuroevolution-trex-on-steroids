import numpy.random as rnd

class MonkeyAgent:
    def __init__(self, num_actions, width, height, path, writer=None):
        self.num_actions = num_actions

    def act(self, state):
        """
        :return: a random action
        """
        return rnd.randint(self.num_actions)
