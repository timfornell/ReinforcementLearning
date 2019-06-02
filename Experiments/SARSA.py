import gym
import sys
import argparse
import numpy as np

import Experiments.BaseEnvironment as gym_env
from Experiments.createParameterDict import createParameterDict

REQUIRED_PARAMS_FROM_ENVIRONMENT = ["Q", "latest_reward", "current_state", "previous_state", "latest_action",
                                    "latest_reward"]
FUNCTION_SPECIFIC_PARAMS = ["gamma", "alpha"]


def SARSA(parameters):
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
    learned_value = reward + gamma * Q_matrix[current_state, new_action] - Q_matrix[previous_state, action]
    Q_matrix[previous_state, action] = Q_matrix[previous_state, action] + alpha * learned_value

    return Q_matrix

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("alpha", help="Factor between [0, 1] which determines the learning rate for the algorithm.")
    p.add_argument("gamma", help="Factor between [0, 1] which determines how important future rewards are.")
    p.add_argument("simulation_environment", help="Which environment that should be used.")
    p.add_argument("episodes", help="Number of episodes to train for.")
    p.add_argument("max_steps", help="Maximum number of actions to take before episode ends.")
    p.add_argument("action_policy", help="Which action policy that should be used.")
    p.add_argument("stochastic", help="If the probability of transitioning from a state to another should be stochastic.",
                   action="store_true")
    p.add_argument("--debug", help="If debug messages should be written, default=False", action="store_true")
    a = p.parse_args()

    env = gym.make(a.simulation_environment)
    probabilities = np.zeros((env.action_space.n, 1))
    if a.stochastic:
        probabilities = input("Please specify the transition probability for all %d actions in the form 'p_1, p_2':\n")

    env_params = {"episodes": int(a.episodes), "max_steps": int(a.max_steps), "stochastic": False,
                  "probabilities": probabilities}
    required_parameters = createParameterDict(REQUIRED_PARAMS_FROM_ENVIRONMENT)
    specific_params = {"alpha": float(a.alpha), "gamma": float(a.gamma)}
    epsilon = 0.5

    if "epsilon_greedy" in a.action_policy:
        epsilon = input("Please specify the value for epsilon: \n")
        
    environment = gym_env.GymEnvironment(env, env_params, QLearningAlgorithm, required_parameters, specific_params,
                                         a.action_policy, {"epsilon": epsilon}, a.debug)
    environment.train()
    environment.plot_results(a.simulation_environment)
    input("Start evaluation")
    environment.evaluate()