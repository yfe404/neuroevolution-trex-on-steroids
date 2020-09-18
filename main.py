import numpy as np
from preprocessor import Preprocessor, SimplePreprocessor
from environment import Environment
from monkey import MonkeyAgent
from neuralnet import NeuralNetAgent
from copy import copy

## Constants
width = 80
height = 80
len_epoch = int(1E8)
num_actions = len(Environment.actions)

def softmax(x):
    return x/x.sum()

def crossover(parentA, parentB):
    ctx0 = np.random.randint(0, parentA.weights_0_1.size)
    ctx1 = np.random.randint(0, parentA.weights_1_2.size)

    offspring = NeuralNetAgent(parentA.num_actions, parentA.input_size, parentA.hidden_size)

    offspring.weights_0_1 = np.concatenate((np.ravel(parentA.weights_0_1)[:ctx0], np.ravel(parentB.weights_0_1)[ctx0:])).reshape(parentB.weights_0_1.shape)

    offspring.weights_1_2 = np.concatenate((np.ravel(parentA.weights_1_2)[:ctx1], np.ravel(parentB.weights_1_2)[ctx1:])).reshape(parentB.weights_1_2.shape)

    return offspring
                    
def mutate(agent, pmut=.01):
    bit = None
    if np.random.random() < .5:
        bit = np.random.randint(0, agent.weights_0_1.shape[0]), np.random.randint(0, agent.weights_0_1.shape[1])
        agent.weights_0_1[bit] = np.random.random()
    else:
        bit = np.random.randint(0, agent.weights_1_2.shape[0]), np.random.randint(0, agent.weights_1_2.shape[1])
        agent.weights_1_2[bit] = np.random.random()

    return agent

def play_generation(population, env, preprocessor):
    gen_count = 0
    print()
    print("#Gen\tmin\tmax\tmean")
    while True:
        fitness = np.array([0 for _ in range(len(population))])
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

        new_generation = []
        new_generation.append(copy(population[np.argmax(fitness)]))

        while len(new_generation) != len(population):
        
            parentA = np.random.choice(population, p=softmax(fitness/fitness.max()))
            parentB = np.random.choice(population, p=softmax(fitness/fitness.max()))
            offspring = mutate(crossover(parentA, parentB))
            new_generation.append(offspring)
        population = new_generation



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
    env = Environment("127.0.0.1", 8765, debug=False)

    preprocessor = SimplePreprocessor()
    #play(agent, env, preprocessor)
    play_generation([NeuralNetAgent(num_actions, input_size=4, hidden_size=5) for _ in range(5)], env, preprocessor)

if __name__ == "__main__":
    main()
