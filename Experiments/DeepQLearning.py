import gym
import sys
import argparse
import torch
import numpy as np
import BaseEnvironment as gym_env
import matplotlib.pyplot as plt

from discretisizeEnvironments import *
from IPython.display import clear_output
from torch.autograd import Variable
import torchvision.transforms as T


class DeepQLearning:

    def __init__(self, env, function_params, env_params):
        self.criterion = torch.nn.MSELoss()
        state_dim = function_params["state_dim"]
        hidden_dim = function_params["hidden_dim"]
        action_dim = function_params["action_dim"]
        self.model = torch.nn.Sequential(torch.nn.Linear(state_dim, hidden_dim),
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Linear(hidden_dim, hidden_dim*2),
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Linear(hidden_dim*2, action_dim))
                                            
        self.optimizer = torch.optim.Adam(self.model.parameters(), function_params["alpha"])
        
    def run_algorithm(self, reward, action, current_state, previous_state):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(previous_state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_policy(self, env):
        pass

    def get_action_from_policy(self, state):
        pass
    
    def get_data_to_save(self, env):
        pass

    def set_variables_from_data(self, data):
        pass

    def plot_result(self, values, title=''):
        ''' Plot the reward curve and histogram of results over time.'''
        # Update the window after each episode
        clear_output(wait=True)
        
        # Define the figure
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
        f.suptitle(title)
        ax[0].plot(values, label='score per run')
        ax[0].axhline(195, c='red',ls='--', label='goal')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Reward')
        x = range(len(values))
        ax[0].legend()
        # Calculate the trend
        try:
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            ax[0].plot(x,p(x),"--", label='trend')
        except:
            print('')
        
        # Plot the histogram of results
        ax[1].hist(values[-50:])
        ax[1].axvline(195, c='red', label='goal')
        ax[1].set_xlabel('Scores per Last 50 Episodes')
        ax[1].set_ylabel('Frequency')
        ax[1].legend()
        plt.show()
