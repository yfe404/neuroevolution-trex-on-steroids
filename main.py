import numpy as np
from preprocessor import Preprocessor, SimplePreprocessor
from environment import Environment
from monkey import MonkeyAgent
from neuralnet import NeuralNetAgent


## Constants
width = 80
height = 80
len_epoch = int(1E8)
num_actions = len(Environment.actions)

def softmax(x):
    return np.array(x)/sum(x)

def crossover(parentA, parentB):
    ctx0 = np.random.randint(0, parentA.weights_0_1.size)
    ctx1 = np.random.randint(0, parentA.weights_1_2.size)

    offspring = NeuralNetAgent(parentA.num_actions, parentA.input_size, parentA.hidden_size)

    offspring.weights_0_1 = np.concatenate((np.ravel(parentA.weights_0_1)[:ctx0], np.ravel(parentB.weights_0_1)[ctx0:])).reshape(parentB.weights_0_1.shape)

    offspring.weights_1_2 = np.concatenate((np.ravel(parentA.weights_1_2)[:ctx1], np.ravel(parentB.weights_1_2)[ctx1:])).reshape(parentB.weights_1_2.shape)

    return offspring
                    
def mutate(agent, pmut=.01):
    mask0 = np.random.random(agent.weights_0_1.shape)
    mask0 = mask0 * (mask0 < pmut)
    agent.weights_0_1 = (agent.weights_0_1 * mask0!=0) * mask0 + (agent.weights_0_1 * mask0==0) * agent.weights_0_1

    mask1 = np.random.random(agent.weights_1_2.shape)
    mask1 = mask1 * (mask1 < pmut)
    agent.weights_1_2 = (agent.weights_1_2 * mask1!=0) * mask1 + (agent.weights_1_2 * mask1==0) * agent.weights_1_2

    return agent

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
        offspring = mutate(crossover(parentA, parentB))
        population[np.argmin(fitness)] = offspring



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
    play_generation([NeuralNetAgent(num_actions, input_size=6, hidden_size=4) for _ in range(3)], env, preprocessor)

if __name__ == "__main__":
    main()
