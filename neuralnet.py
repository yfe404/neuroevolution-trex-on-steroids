import numpy as np

class NeuralNetAgent:
    def __init__(self, num_actions, input_size, hidden_size):
        self.num_actions = num_actions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights_0_1 = 2 * np.random.random((input_size, hidden_size)) - 1
        self.weights_1_2 = 2 * np.random.random((hidden_size, num_actions)) - 1

    def act(self, state):
        """
        :return: predicted  action
        """

        def relu(x):
            return (x > 0) * x

        layer_0 = state
        layer_1 = relu(np.dot(layer_0, self.weights_0_1))
        layer_2 = np.dot(layer_1, self.weights_1_2)

        return layer_2.argmax()
