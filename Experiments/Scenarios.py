import gym
import colorama
import SARSA, QLearning, ExpectedSARSA, BaseEnvironment

# DO NOT REMOVE - Is needed to print lovely colors in bash
colorama.init()

""" 
*******************************************************
                      QLearning
*******************************************************
"""
# CliffWalking-v0
def QLearning_RunCliffWalking(load_previous_training, debug):
    simulation_environment = "CliffWalking-v0"
    episodes = 1000
    max_steps = 100

    action_policy = "epsilon_greedy"
    epsilon = 0.5
    alpha = 0.5
    gamma = 0.3

    env = gym.make(simulation_environment)
    env_params = {"episodes": episodes, "max_steps": max_steps, "stochastic": False}

    function_specific_params = {"alpha": alpha, "gamma": gamma, "qInit": "stochastic"}
    action_policy_params = {"epsilon": epsilon}
    
    QLearning_object = QLearning.QLearning(env, function_specific_params, env_params)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, QLearning_object, action_policy, action_policy_params, debug)

    gym_environment.train()

    if load_previous_training:
        gym_environment.evaluate(simulation_environment)
    else:
        gym_environment.evaluate()

    gym_environment.plot_results(simulation_environment)

# Taxi-v3
def QLearning_RunTaxi(load_previous_training, debug):
    simulation_environment = "Taxi-v3"
    episodes = 1000
    max_steps = 200

    action_policy = "epsilon_greedy"
    epsilon = 0.9
    alpha = 0.5
    gamma = 0.3

    env = gym.make(simulation_environment)
    env_params = {"episodes": episodes, "max_steps": max_steps, "stochastic": False}

    function_specific_params = {"alpha": alpha, "gamma": gamma, "qInit": "stochastic"}
    action_policy_params = {"epsilon": epsilon}
    
    QLearning_object = QLearning.QLearning(env, function_specific_params, env_params)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, QLearning_object, action_policy, action_policy_params, debug)

    gym_environment.train()
    
    if load_previous_training:
        gym_environment.evaluate(simulation_environment)
    else:
        gym_environment.evaluate()

    gym_environment.plot_results(simulation_environment)

# MountainCar-v0
def QLearning_RunMountainCar(load_previous_training, debug):
    simulation_environment = "MountainCar-v0"
    episodes = 10000
    max_steps = 200

    action_policy = "epsilon_greedy_update"
    epsilon = 0.9
    alpha = 0.6
    gamma = 0.9

    env = gym.make(simulation_environment)
    # Parameters to use when discretizing the environment
    disc_params = {"resolution": [0.01, 0.01]}
    env_params = {"episodes": episodes, "max_steps": max_steps, "stochastic": False, "disc_param": disc_params}

    function_specific_params = {"alpha": alpha, "gamma": gamma, "qInit": "stochastic"}
    action_policy_params = {"epsilon": epsilon}
    
    QLearning_object = QLearning.QLearning(env, function_specific_params, env_params)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, QLearning_object, action_policy, action_policy_params, debug)

    gym_environment.train()
    
    if load_previous_training:
        gym_environment.evaluate(simulation_environment)
    else:
        gym_environment.evaluate()
    
    gym_environment.plot_results(simulation_environment)

""" 
*******************************************************
                        SARSA
*******************************************************
"""
# CliffWalking-v0
def SARSA_RunCliffWalking(load_previous_training, debug):
    simulation_environment = "CliffWalking-v0"
    episodes = 1000
    max_steps = 100

    action_policy = "epsilon_greedy"
    epsilon = 0.5
    alpha = 0.5
    gamma = 0.3

    env = gym.make(simulation_environment)
    env_params = {"episodes": episodes, "max_steps": max_steps, "stochastic": False}

    function_specific_params = {"alpha": alpha, "gamma": gamma, "qInit": "stochastic"}
    action_policy_params = {"epsilon": epsilon}
    
    SARSA_Object = SARSA.SARSA(env, function_specific_params)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, SARSA_Object, action_policy, action_policy_params, debug)

    gym_environment.train()
    
    if load_previous_training:
        gym_environment.evaluate(simulation_environment)
    else:
        gym_environment.evaluate()

    gym_environment.plot_results(simulation_environment)        

""" 
*******************************************************
                    Expected SARSA
*******************************************************
"""
# CliffWalking-v0
def ExpectedSARSA_RunCliffWalking(load_previous_training, debug):
    simulation_environment = "CliffWalking-v0"
    episodes = 1000
    max_steps = 1000

    action_policy = "epsilon_greedy_update"
    epsilon = 0.9
    alpha = 0.1
    gamma = 0.9

    env = gym.make(simulation_environment)
    env_params = {"episodes": episodes, "max_steps": max_steps, "stochastic": False}

    function_specific_params = {"alpha": alpha, "gamma": gamma, "qInit": "stochastic"}
    action_policy_params = {"epsilon": epsilon}
    
    ExpectedSARSA_Object = ExpectedSARSA.ExpectedSARSA(env, function_specific_params)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, ExpectedSARSA_Object, action_policy, action_policy_params, debug)

    gym_environment.train()
    
    if load_previous_training:
        gym_environment.evaluate(simulation_environment)
    else:
        gym_environment.evaluate()
    
    gym_environment.plot_results(simulation_environment)
