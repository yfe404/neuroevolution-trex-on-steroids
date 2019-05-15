import numpy as np
from preprocessor import Preprocessor
from environment import Environment
from monkey import MonkeyAgent
from cnn_agent import CNNAgent

## Constants
width = 80
height = 80
len_epoch = int(1E8)
num_actions = len(Environment.actions)

def softmax(x):
    return np.array(x)/sum(x)

def crossover(parentA, parentB):
    offspring = CNNAgent(num_actions, width, height, None, None)

    for idx, layer in enumerate(parentB.model.model.layers):
        w = layer.get_weights()
        if len(w) > 0:
            pos_0 = np.random.randint(np.size(w[0]))
            pos_1 = np.random.randint(np.size(w[1]))
            shape_0 = w[0].shape
            shape_1 = w[1].shape
            
            w_in = np.concatenate((
                w[0].ravel()[:pos_0],
                parentA.model.model.layers[idx].get_weights()[0].ravel()[pos_0:]
            ))
            w_out = np.concatenate((
                w[1].ravel()[:pos_1],
                parentA.model.model.layers[idx].get_weights()[1].ravel()[pos_1:]
            ))
            
            offspring.model.model.layers[idx].set_weights([
                w_in.reshape(shape_0),
                w_out.reshape(shape_1)
            ])
        return offspring
                
def mutate(agent, pmut=.01):
    for idx, layer in enumerate(agent.model.model.layers):
        w = layer.get_weights()
        if len(w) > 0:
            for x in np.nditer(w[0], op_flags=['readwrite']):
                if np.random.random() < pmut:
                    x[...] = np.random.normal(0, 1)
            for x in np.nditer(w[1], op_flags=['readwrite']):
                if np.random.random() < pmut:
                    x[...] = np.random.normal(0, 1)

            agent.model.model.layers[idx].set_weights([w[0], w[1]])

def play_generation(population, env, preprocessor):
    gen_count = 0
    print()
    print("#Gen\tmin\tmax\tmean")
    while True:
        fitness = [0 for _ in range(len(population))]
        for idx,agent in enumerate(population):
            frame, _, crashed = env.start_game()
            frame = preprocessor.process(frame)
            state = preprocessor.get_initial_state(frame)

            while not crashed:
                action = agent.act(state)
                next_frame, reward, crashed = env.do_action(action)
                #print("action: {}".format(env.actions[action]))
                next_frame = preprocessor.process(next_frame)
                next_state = preprocessor.get_updated_state(next_frame)
                fitness[idx] += reward
                state = next_state


        #print(fitness)
        print(f"{gen_count}\t{min(fitness)}\t{max(fitness)}\t{np.mean(fitness)}")
        
        gen_count += 1
        #print(softmax(fitness))
        parentA = np.random.choice(population, p=softmax(fitness))
        parentB = np.random.choice(population, p=softmax(fitness))
        offspring = crossover(parentA, parentB)
        population[np.argmin(fitness)] = offspring

        for agent in population:
            mutate(agent)



def play(agent, env, preprocessor):
    while True:
        frame, _, crashed = env.start_game()
        frame = preprocessor.process(frame)
        state = preprocessor.get_initial_state(frame)

        while not crashed:
            action = agent.act(state)
            next_frame, reward, crashed = env.do_action(action)
            print("action: {}".format(env.actions[action]))
            next_frame = preprocessor.process(next_frame)
            next_state = preprocessor.get_updated_state(next_frame)

            state = next_state

        #print("Crash")

def main():
    # Initialize key objects: environment, agent and preprocessor
    env = Environment("127.0.0.1", 8765)
    agent = CNNAgent(num_actions, width, height, None, None)

    preprocessor = Preprocessor(width, height)
    #play(agent, env, preprocessor)
    play_generation([CNNAgent(num_actions, width, height, None, None) for _ in range(10)], env, preprocessor)

if __name__ == "__main__":
    main()
