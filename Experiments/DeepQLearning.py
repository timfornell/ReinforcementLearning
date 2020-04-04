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
        self._criterion = torch.nn.MSELoss()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        hidden_dim = function_params["hidden_dim"]
        self._model = torch.nn.Sequential(torch.nn.Linear(state_dim, hidden_dim),
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Linear(hidden_dim, hidden_dim*2),
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Linear(hidden_dim*2, action_dim))

        self._alpha = function_params["alpha"]
        self._gamma = function_params["gamma"]                                          
        self._optimizer = torch.optim.Adam(self._model.parameters(), self._alpha)
        self._memory = []
        self._double = function_params["double"]
        self._soft = function_params["soft"]
        self._n_update = function_params["n_update"]
        self._replay = function_params["replay"]
        if self._replay:
            self._replay_size = function_params["replay_size"]
        
    def run_algorithm(self, reward, action, current_state, previous_state, done):
        q_values = self.predict(previous_state).tolist()

        if done and not self._replay:
            q_values[action] = reward
            # Update network weights
            self.update(previous_state, q_values)
        
        if not done:
            if self._replay:
                # Update network weights using replay memory
                self.replay(self._memory, self._replay_size, self._gamma)
            else:
                # Update network weights using the last step only
                q_values_next = self.predict(current_state)
                q_values[action] = reward + self._gamma * torch.max(q_values_next).item()
                self.update(previous_state, q_values)
    
    def update(self, previous_state, q_values):
        """Update the weights of the network given a training sample. """
        y_pred = self._model(torch.Tensor(previous_state))
        loss = self._criterion(y_pred, Variable(torch.Tensor(q_values)))
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
    
    def predict(self, current_state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self._model(torch.Tensor(current_state))

    def run_prerequisite(self, episode):
        if self._double and not self._soft:
            # Update target network every n_update steps
            if episode % self._n_update == 0:
                self._model.target_update()
        if self._double and self._soft:
            self._model.target_update()
    
    def get_action(self, state):
        q_values = self.predict(state)
        action = torch.argmax(q_values).item()
        return action

    def save_data_from_action(self, previous_state, action, new_state, reward, done):
        self._memory.append((previous_state, action, new_state, reward, done))

    def run_post_episode_updates(self):
        pass

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
