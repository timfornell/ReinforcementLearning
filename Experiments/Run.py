import gym
import SARSA, QLearning, ExpectedSARSA, createParameterDict, BaseEnvironment

""" 
*******************************************************
                      QLearning
*******************************************************
"""
# CliffWalking-v0
run = True
if run is True:
    # simulation_environment = "CliffWalking-v0"
    episodes = 1000
    max_steps = 100
    debug = False

    action_policy = "epsilon_greedy"
    epsilon = 0.5
    alpha = 0.5
    gamma = 0.3

    env = gym.make(simulation_environment)
    env_params = {"episodes": episodes, "max_steps": max_steps, "stochastic": False}

    function_specific_params = {"alpha": alpha, "gamma": gamma}
    action_policy_params = {"epsilon": epsilon}
    required_params = createParameterDict.createParameterDict(QLearning.REQUIRED_PARAMS_FROM_ENVIRONMENT)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, QLearning.QLearningAlgorithm, 
                                                     required_params, function_specific_params, action_policy,
                                                     action_policy_params, debug)

    gym_environment.train()
    input("Training finished, press enter to start evaluation and plot results.")
    gym_environment.evaluate()
    gym_environment.plot_results(simulation_environment)

""" 
*******************************************************
                        SARSA
*******************************************************
"""
# CliffWalking-v0
run = False
if run is True:
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

    function_specific_params = {"alpha": alpha, "gamma": gamma}
    action_policy_params = {"epsilon": epsilon}
    required_params = createParameterDict.createParameterDict(SARSA.REQUIRED_PARAMS_FROM_ENVIRONMENT)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, SARSA.SARSA, 
                                                    required_params, function_specific_params, action_policy,
                                                    action_policy_params, debug)

    gym_environment.train()
    input("Training finished, press enter to start evaluation and plot results.")
    gym_environment.evaluate()
    gym_environment.plot_results(simulation_environment)    

""" 
*******************************************************
                    Expected SARSA
*******************************************************
"""
# CliffWalking-v0
run = False
if run is True:
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

    function_specific_params = {"alpha": alpha, "gamma": gamma}
    action_policy_params = {"epsilon": epsilon}
    required_params = createParameterDict.createParameterDict(ExpectedSARSA.REQUIRED_PARAMS_FROM_ENVIRONMENT)

    gym_environment = BaseEnvironment.GymEnvironment(env, env_params, ExpectedSARSA.ExpectedSARSA, 
                                                    required_params, function_specific_params, action_policy,
                                                    action_policy_params, debug)

    gym_environment.train()
    input("Training finished, press enter to start evaluation and plot results.")
    gym_environment.evaluate()
    gym_environment.plot_results(simulation_environment)