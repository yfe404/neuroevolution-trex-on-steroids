import numpy as np
from preprocessor import Preprocessor
from environment import Environment
from monkey import MonkeyAgent


## Constants
width = 80
height = 80
len_epoch = int(1E8)
num_actions = len(Environment.actions)

def softmax(x):
    return np.array(x)/sum(x)

def crossover(parentA, parentB):
    raise NotImplementedError
                    
def mutate(agent, pmut=.01):
    raise NotImplementedError

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
    agent = MonkeyAgent(num_actions, width, height, None, None)

    preprocessor = Preprocessor(width, height)
    #play(agent, env, preprocessor)
    play_generation([MonkeyAgent(num_actions, width, height, None, None) for _ in range(10)], env, preprocessor)

if __name__ == "__main__":
    main()
