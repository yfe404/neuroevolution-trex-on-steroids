"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import signal
import neat
import visualize
import time
import multiprocessing
import subprocess
import numpy as np
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler

from preprocessor import Preprocessor, SimplePreprocessor
from environment import Environment

import webbrowser
from multiprocessing import Lock

def run_http(port, server_class=HTTPServer, handler_class=BaseHTTPRequestHandler):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()


class Process(multiprocessing.Process): 
    def __init__(self, id): 
        super(Process, self).__init__() 
        self.id = id
                 
    def run(self): 
        time.sleep(1) 
        print("I'm the process with id: {}".format(self.id)) 


p8888 = Lock() # 8765
p8889 = Lock() # 8766
p8890 = Lock() # 8767
p8891 = Lock() # 8768
p8892 = Lock() # 8769
#
p8893 = Lock() # 8770
p8894 = Lock() # 8771
p8895 = Lock() # 8772
p8896 = Lock() # 8773
p8897 = Lock() # 8774

def get_port():
    if p8888.acquire(block=False):
        return 0

    if p8889.acquire(block=False):
        return 1

    if p8890.acquire(block=False):
        return 2

    if p8891.acquire(block=False):
        return 3

    if p8892.acquire(block=False):
        return 4

    if p8893.acquire(block=False):
        return 5

    if p8894.acquire(block=False):
        return 6

    if p8895.acquire(block=False):
        return 7

    if p8896.acquire(block=False):
        return 8

    if p8897.acquire(block=False):
        return 9


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome[1], config)
    fitness = 0.0

    _id = get_port()
    print(_id)

    env = Environment("127.0.0.1", 8765+_id, debug=False)
    time.sleep(2)

    subprocess.Popen(["qutebrowser", "--target", "window", f"http://localhost:{8888+_id}"])

    preprocessor = SimplePreprocessor()

    time.sleep(10)
    frame, _, crashed = env.start_game()
    frame = preprocessor.process(frame)
    state = preprocessor.get_initial_state(frame)
    
    while not crashed:
        action = np.argmax(net.activate(state))
        next_frame, reward, crashed = env.do_action(action)
        #print("action: {}".format(env.actions[action]))
        next_frame = preprocessor.process(next_frame)
        next_state = preprocessor.get_updated_state(next_frame)
        fitness += reward
        state = next_state

    env.close()

    time.sleep(2)

    return fitness
    
def eval_genomes(genomes, config):
    pool = multiprocessing.Pool(processes=10)
    eval_genome_fixed_conf=partial(eval_genome, config=config)
    fitnesses = pool.map(eval_genome_fixed_conf, genomes)
    for genome, fitness in zip([g[1] for g in genomes], fitnesses):
        genome.fitness = fitness

    subprocess.Popen(["pkill", "qutebrowser"])
    p8888.release()
    p8889.release()
    p8890.release()
    p8891.release()
    p8892.release()

    p8893.release()
    p8894.release()
    p8895.release()
    p8896.release()
    p8897.release()

    
def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 3000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    #node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    #visualize.draw_net(config, winner, True, node_names=node_names)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
