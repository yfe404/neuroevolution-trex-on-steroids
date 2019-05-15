from copy import deepcopy
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten

class CNN:
    def __init__(self, height, width, num_actions):
        self.height = height
        self.width = width
        self.num_actions = num_actions
        
        input_shape = (4, height, width)
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(8,8), strides=(4,4),
                  activation='relu',
                  padding='same',
                  input_shape=input_shape))
#        self.model.add(Conv2D(64, kernel_size=(4,4), strides=(2,2),
#                         padding='same',
#                         activation='relu'))
        self.model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1),
                         padding='same',
                         activation='relu'))
        self.model.add(Flatten())
#        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(num_actions, init='uniform', activation='linear'))

        self.model.compile(loss='mse', optimizer='adam')

        for idx, layer in enumerate(self.model.layers):
            w = layer.get_weights()
            if len(w) > 0: ## prevent accessing in and out weights which obvisouly doesn't exist on the flatten
                w_in = np.random.normal(0, 1, w[0].shape)
                w_out = np.random.normal(0, 1, w[1].shape)
                self.model.layers[idx].set_weights([w_in, w_out])

    def predict(self, state):
        state = state.reshape(-1, 4, 80, 80)
        action = self.model.predict(state)
        action = np.argmax(action)
        
        return action


    def clone(self):
        clonie = CNN(self.height, self.width, self.num_actions)
        for idx,layer in enumerate(self.model.layers):
            clonie.model.layers[idx].set_weights(deepcopy(layer.get_weights()))
    
        return clonie
