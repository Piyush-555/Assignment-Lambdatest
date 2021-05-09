import os
import torch
import numpy as np
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

import NN_ES as NN
from simple_es import Simple_ES


def multiple_tries(model, soln):
    NN.load_param(model, soln)
    num_tries = 4

    pool = mp.Pool(processes=max(4, num_tries))
    rewards = pool.starmap(NN.fitness_function, [[model, False] for i in range(num_tries)])
    pool.close()

    return -np.mean(rewards).item()


if __name__ == '__main__':
    generations = 500
    eps = 0.9
    model = NN.NN_v2()
    es = Simple_ES(NN.param2numpy(model), sigma=0.1)

    mean = []
    best = []
    for gen in range(1, generations + 1):
        print("Generation:: {}..".format(gen))
        
        solutions = es.ask()
        function_values = [multiple_tries(model, s) for s in solutions]
        es.tell(solutions, function_values)

        mean.append(-np.mean(function_values))
        best.append(-min(function_values))
        print("Mean: {:.2f}\tBest:{}\n".format(mean[-1], best[-1]))

        # Decaying sigma
        if gen%10 == 0:
            es.sigma = es.sigma * eps
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7, 4), dpi=300)
    ax = plt.gca()

    plt.plot(list(range(1, generations+1)), mean, label="Mean Reward")
    plt.plot(list(range(1, generations+1)), best, label="Best Reward")

    ax.set_ylabel('Reward')
    ax.set_xlabel('num-generation')
    plt.title("Cartpole - Simple ES")
    ax.legend()

    fig.savefig("plot.png", bbox_inches='tight')
    plt.close()
