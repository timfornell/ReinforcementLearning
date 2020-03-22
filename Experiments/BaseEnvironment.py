"""

This file is intended to be used as a base environment for different reinforcemeant learning algorithms. It is meant to
be resposible for performing the training, evaluating and plotting the results.

"""

import sys
import gym
import time
import pprint
import numpy as np
import matplotlib.pyplot as plt

import MovingAverage as moving_average

AVAILABLE_ENVIRONMENT_PARAMS = ["episodes", "max_steps", "stochastic", "probabilities"]
EVALUATE = "eval"


class GymEnvironment:

    def __init__(self, env, env_params: dict, RL_class_object, action_policy: str, action_policy_params: dict, debug: bool):
        for param in env_params:
            if param not in AVAILABLE_ENVIRONMENT_PARAMS:
                sys.exit("Cannot find parameter: %s, exiting..." % param)

        self._env = env
        self._env_params = env_params
        self._RL_class_object = RL_class_object
        self._debug = debug
        self._action_policy = action_policy
        self._action_policy_params = action_policy_params
        self._previous_state = 0
        self._current_state = 0
        self._latest_reward = 0
        self._latest_action = 0
        self._reward_per_episode = np.zeros((env_params["episodes"], 1))

    def train(self):
        self.setup_environment()

        for ep in range(self._env_params["episodes"]):
            print("Training episode %d" % ep)
            self._current_state = self._env.reset()

            # Count initial state
            self._RL_class_object._visited_states[self._current_state] += 1

            num_steps = 0
            for steps in range(self._env_params["max_steps"]):
                # Decide which action to take
                self._latest_action = self.get_action(episode=ep)

                # Perform action and update state and reward variables
                done = self.perform_action(self._latest_action, episode=ep)

                # Run Reinforcement Learning (RL) Algorithm to update Q matrix
                self._RL_class_object.run_algorithm(self._latest_reward, self._latest_action, self._current_state, self._previous_state)

                self._RL_class_object.update_policy(self._env)

                num_steps += 1
                if done:
                    break
            
            print("Episode %d ended after %d steps with reward %d" % (ep, num_steps, self._reward_per_episode[ep]))

    def get_action(self, episode):
        action = self._env.action_space.sample()
        Q = self._RL_class_object._Q

        if self._action_policy == "epsilon_greedy":
            if np.random.uniform(low=0, high=1) < 1 - self._action_policy_params["epsilon"]:
                action = np.argmax(Q[self._current_state, :])
            else:
                action = self._env.action_space.sample()
        elif self._action_policy == "epsilon_greedy_update":
            if np.random.uniform(low=0, high=1) < 1 - self._action_policy_params["epsilon"]:
                action = np.argmax(Q[self._current_state, :])
            else:
                action = self._env.action_space.sample()

            self._action_policy_params["epsilon"] = self._action_policy_params["epsilon"] / (episode + 1)
        else:
            sys.exit("Could not find action policy: %s, exiting..." % self._action_policy)

        return action

    def perform_action(self, action, episode):
        new_state, reward, done, info = self._env.step(action)

        self._previous_state = self._current_state
        self._current_state = new_state

        self._reward_per_episode[episode] += reward
        self._latest_reward = reward

        return done

    def setup_environment(self):
        for key, value in self._env_params.items():
            if key == "stochastic" and value is True:
                self.set_transition_probabilites(self._env_params["probabilities"])

    def get_plot_data(self, environment):
        reshape_x = 0
        reshape_y = 0
        if environment == "CliffWalking-v0":
            reshape_x = 4
            reshape_y = 12
        elif environment == "FrozenLake-v0":
            reshape_x = 4
            reshape_y = 4
        else:
            sys.exit("Could not find %s" % environment)

        Q_max = np.zeros((self._env.observation_space.n, 1))
        Q_plot = np.zeros((self._env.observation_space.n, 1))

        for state in range(0, np.size(self._policy)):
            Q_plot[state] = np.max(self._Q[state, :])
            Q_max[state] = np.argmax(self._Q[state, :])

        states_reshaped = self._visited_states.reshape(reshape_x, reshape_y)
        
        return Q_max.reshape(reshape_x, reshape_y), Q_plot.reshape(reshape_x, reshape_y), states_reshaped

    def plot_results(self, environment):
        Q_max, Q_plot, states_reshaped = self.get_plot_data(environment)

        fig1 = plt.figure(figsize=(6, 1))

        fig1.add_subplot(3, 1, 1)
        plt.imshow(Q_plot)
        plt.title("State Action function")
        plt.colorbar()

        fig1.add_subplot(3, 1, 2)
        plt.imshow(Q_max)
        plt.title("Policy")
        plt.colorbar()

        fig1.add_subplot(3, 1, 3)
        plt.imshow(states_reshaped)
        plt.title("number of visits per state")
        plt.colorbar()

        plt.figure(2)
        plt.title("total expected reward")
        plt.clf()
        reward_per_episode_aver = moving_average.MovingAverage(self._reward_per_episode, 10)
        plt.plot(self._reward_per_episode, alpha=0.5)
        plt.plot(reward_per_episode_aver, alpha=0.5)
        plt.legend(["Reward", "Smoothed reward"])
        plt.xlabel("Iterations")
        plt.ylabel("Reward")
        plt.show()

    def set_transition_probabilites(self, probabilities):
        if len(probabilities) < self._env.action_space.n:
            sys.exit("Number of actions is not the same as number of probabilities specified.")
        elif float(sum(probabilities)) != 1.0:
            sys.exit("Total sum of probabilities is not equal to one.")

        for state in range(self._env.observation_space.n):
            for action in range(self._env.action_space.n):
                items = list(self._env.P[state][action][0])
                items[0] = probabilities[action]
                self._env.P[state][action][0] = tuple(items)

    def evaluate(self):
        step = 1
        self._RL_class_object.update_policy(self._env)
        state = self._env.reset()

        done = False
        while not done:
            print("=========\n Step %d\n=========" % step)
            self._env.render()

            action = int(self._RL_class_object.get_action_from_policy(state))
            state, reward, done, info = self._env.step(action)

            step += 1
            time.sleep(0.1)
