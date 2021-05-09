"""
Cartpole environment using Neural Network with training
using Simple Gaussian Evolution Strategy.
"""
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Import dependencies
import os
import gym
import numpy as np
import pandas as pd
from collections import OrderedDict 

import torch
from torch import nn
import torch.nn.functional as F


# Creating util functions

def param2numpy(model):
    params_list = []
    for p in model.parameters():
        params_list.append(p.flatten())
    return torch.cat(params_list, dim=-1).detach().cpu().numpy()


def load_param(model, params):
    nodes_in = 4
    nodes_hidden_1 = 32
    nodes_hidden_2 = 32
    nodes_out = 2

    updated_dict = OrderedDict()
    
    updated_dict["fc1.weight"] = params[:nodes_in * nodes_hidden_1].reshape(nodes_hidden_1, nodes_in)
    till = nodes_in * nodes_hidden_1
    updated_dict["fc1.bias"] = params[till:till + nodes_hidden_1]
    till += nodes_hidden_1
    
    updated_dict["fc2.weight"] = params[till: till + nodes_hidden_1 * nodes_hidden_2].reshape(nodes_hidden_1, nodes_hidden_2)
    till += nodes_hidden_1 * nodes_hidden_2
    updated_dict["fc2.bias"] = params[till:till + nodes_hidden_2]
    till += nodes_hidden_2
    
    updated_dict["out.weight"] = params[till:till + nodes_hidden_2 * nodes_out].reshape(nodes_out, nodes_hidden_2)
    till += nodes_hidden_2 * nodes_out
    updated_dict["out.bias"] = params[till:]

    for k, v in updated_dict.items():
        updated_dict[k] = torch.tensor(v)
    
    model.load_state_dict(updated_dict)

    
def fitness_function(model, render=False):
    env = gym.make('CartPole-v0')
    obs = env.reset()
    total_reward = 0

    while True:
        if render:
            env.render()
        obs = torch.Tensor(obs)
        action = F.softmax(model(obs)).argmax(axis=-1)
        obs, reward, done, _ = env.step(action.squeeze().detach().numpy())
        total_reward += reward
        
        if done:
            env.close()
            break

    # print("Reward:", total_reward)

    return total_reward


# Defining model
class NN_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.out = nn.Linear(32, 2, bias=True)

    def forward(self, inputs):
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        return self.out(out)
