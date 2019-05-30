import gym
import sys
import argparse
import numpy as np

import Experiments.base_environment as gym_env

REQUIRED_PARAMS_FROM_ENVIRONMENT = ["Q", "latest_reward", "current_state", "previous_state", "latest_action",
                                    "latest_reward"]
FUNCTION_SPECIFIC_PARAMS = ["gamma", "alpha"]


def ExpectedSARSA(parameters):
    for param in REQUIRED_PARAMS_FROM_ENVIRONMENT:
        if param not in parameters:
            sys.exit("Cannot perform QLearning without parameter: %s, exiting..." % param)

    reward = parameters["latest_reward"]
    Q_matrix = parameters["Q"]
    current_state = parameters["current_state"]
    previous_state = parameters["previous_state"]
    action = parameters["latest_action"]

    alpha = parameters["alpha"]
    gamma = parameters["gamma"]

    new_action = np.argmax(Q_matrix[current_state, :])
    learned_value = reward + gamma * np.mean(Q_matrix[current_state, :]) - Q_matrix[previous_state, action]
    Q_matrix[previous_state, action] = Q_matrix[previous_state, action] + alpha * learned_value

    return Q_matrix


def createParameterDict(alpha, gamma):
    param_dict = dict()
    for param in REQUIRED_PARAMS_FROM_ENVIRONMENT:
        param_dict[param] = ""

    return param_dict


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("alpha", help="Factor between [0, 1] which determines the learning rate for the algorithm.")
    p.add_argument("gamma", help="Factor between [0, 1] which determines how important future rewards are.")
    p.add_argument("simulation_environment", help="Which environment that should be used.")
    p.add_argument("episodes", help="Number of episodes to train for.")
    p.add_argument("max_steps", help="Maximum number of actions to take before episode ends.")
    p.add_argument("action_policy", help="Which action policy that should be used.")
    p.add_argument("--debug", help="If debug messages should be written, default=False", action="store_true")
    a = p.parse_args()

    env = gym.make(a.simulation_environment)
    env_params = {"episodes": int(a.episodes), "max_steps": int(a.max_steps), "stochastic": False,
                  "probabilities": [0.85, 0.05, 0.05, 0.05]}
    required_parameters = createParameterDict(a.alpha, a.gamma)
    specific_params = {"alpha": float(a.alpha), "gamma": float(a.gamma)}
    epsilon = 0.3

    environment = gym_env.GymEnvironment(env, env_params, ExpectedSARSA, required_parameters, specific_params,
                                         a.action_policy, {"epsilon": epsilon}, a.debug)
    environment.train()
    environment.plot_results(a.simulation_environment)