from cnn import CNN

import numpy.random as rnd

class CNNAgent:
    def __init__(self, num_actions, width, height, path, writer):
        self.num_actions = num_actions
        self.model = CNN(height, width, num_actions)

    def act(self, state):
        """
        :return: a random action
        """
        return self.model.predict(state)
    
#        return rnd.randint(self.num_actions)

        
