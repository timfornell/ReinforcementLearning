import gym
import colorama
import SARSA, QLearning, ExpectedSARSA, createParameterDict, BaseEnvironment

# DO NOT REMOVE - Is needed to print lovely colors in bash
colorama.init()

""" 
*******************************************************
                      QLearning
*******************************************************
"""
# CliffWalking-v0
def QLearning_RunCliffWalking():
    simulation_environment = "CliffWalking-v0"
    episodes = 1000
    max_steps = 100
    debug = False

    action_policy = "epsilon_greedy"
    epsilon = 0.5
    alpha = 0.5
    gamma = 0.3

    env = gym.make(simulation_environment)
    env_params = {"episodes": episodes, "max_steps": max_steps, "stochastic": False}

    function_specific_params = {"alpha": alpha, "gamma": gamma, "qInit": "stochastic"}
    action_policy_params = {"epsilon": epsilon}
    
    QLearning_object = QLearning.QLearning(env, function_specific_params)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, QLearning_object, action_policy, action_policy_params, debug)

    gym_environment.train()
    input("Training finished, press enter to start evaluation and plot results.")
    gym_environment.evaluate()
    gym_environment.plot_results(simulation_environment)

# Taxi-v3
def QLearning_RunTaxi():
    simulation_environment = "Taxi-v3"
    episodes = 1000
    max_steps = 200
    debug = False

    action_policy = "epsilon_greedy"
    epsilon = 0.9
    alpha = 0.5
    gamma = 0.3

    env = gym.make(simulation_environment)
    env_params = {"episodes": episodes, "max_steps": max_steps, "stochastic": False}

    function_specific_params = {"alpha": alpha, "gamma": gamma, "qInit": "stochastic"}
    action_policy_params = {"epsilon": epsilon}
    
    QLearning_object = QLearning.QLearning(env, function_specific_params)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, QLearning_object, action_policy, action_policy_params, debug)

    gym_environment.train()
    gym_environment.evaluate(simulation_environment)
    gym_environment.plot_results(simulation_environment)

""" 
*******************************************************
                        SARSA
*******************************************************
"""
# CliffWalking-v0
def SARSA_RunCliffWalking():
    simulation_environment = "CliffWalking-v0"
    episodes = 1000
    max_steps = 100
    debug = False

    action_policy = "epsilon_greedy"
    epsilon = 0.5
    alpha = 0.5
    gamma = 0.3

    env = gym.make(simulation_environment)
    env_params = {"episodes": episodes, "max_steps": max_steps, "stochastic": False}

    function_specific_params = {"alpha": alpha, "gamma": gamma, "qInit": "stochastic"}
    action_policy_params = {"epsilon": epsilon}
    
    QLearning_object = QLearning.QLearning(env, function_specific_params)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, QLearning_object, action_policy, action_policy_params, debug)

    gym_environment.train()
    gym_environment.evaluate(simulation_environment)
    gym_environment.plot_results(simulation_environment)  

""" 
*******************************************************
                    Expected SARSA
*******************************************************
"""
# CliffWalking-v0
def ExpectedSARSA_RunCliffWalking():
    simulation_environment = "CliffWalking-v0"
    episodes = 1000
    max_steps = 100
    debug = False

    action_policy = "epsilon_greedy_update"
    epsilon = 0.8
    alpha = 0.8
    gamma = 0.9

    env = gym.make(simulation_environment)
    env_params = {"episodes": episodes, "max_steps": max_steps, "stochastic": False}

    function_specific_params = {"alpha": alpha, "gamma": gamma, "qInit": "stochastic"}
    action_policy_params = {"epsilon": epsilon}
    
    QLearning_object = QLearning.QLearning(env, function_specific_params)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, QLearning_object, action_policy, action_policy_params, debug)

    gym_environment.train()
    gym_environment.evaluate(simulation_environment)
    gym_environment.plot_results(simulation_environment)


# QLearning_RunCliffWalking()
# QLearning_RunTaxi()
# SARSA_RunCliffWalking()
# ExpectedSARSA_RunCliffWalking()