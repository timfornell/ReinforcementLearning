import numpy as np

CONTINUOUS_ENVIRONMENTS = ["MountainCar-v0"]

def discretisize_environment(env, params, stochastic):
    Q_vector = np.array((1,1))
    state_vector = np.array((1,1))
    if env.spec.id == "MountainCar-v0":
        Q_vector, state_vector = discretisize_mountain_car(env, params, stochastic)
    else:
        pass

    return Q_vector, state_vector


def discretisize_mountain_car(env, params, stochastic):
    lowest_values = env.observation_space.low
    highest_values = env.observation_space.high
    resolution = params["disc_param"]["resolution"]

    num_dimensions = np.size(lowest_values)
    num_states = []
    
    for i in range(0, num_dimensions):
        discrete_states = np.floor((highest_values[i] - lowest_values[i]) / resolution[i])
        if discrete_states <= 1:
            input("There is only 1 state in one of the dimensions, press enter if you want to continue.")

        num_states.append(discrete_states)
    
    num_states = [int(f) for f in num_states]
    state_vector = np.zeros(num_states)
    
    num_states.append(env.action_space.n)
    if stochastic:
        Q_vector = np.random.uniform(0, 1, num_states)
    else:
        Q_vector = np.zeros(num_states)

    return Q_vector, state_vector


def convert_state_to_discrete(env, params, state):
    lowest_values = env.observation_space.low
    resolution = params["disc_param"]["resolution"]
    inv_resolution = [1 / f for f in resolution]

    state_shifted = state - lowest_values
    state_index = np.round(state_shifted * np.array(inv_resolution), 0).astype(int)
    
    return state_index