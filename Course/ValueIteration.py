import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CliffWalking-v0')

numActions = 4

def valueIteration(gamma, theta, iteration, delta):
    V = np.zeros((env.observation_space.n, 1))
    policy = np.zeros((env.observation_space.n, 1))
    init = env.reset()

    while delta > theta:
        delta = 0
        actionRewardArray = np.zeros((4, 1))

        for state in range(env.observation_space.n):
            print("State {}".format(state))
            stateValue = float(V[state])

            for action in range(numActions):
                prob = env.P[state][action][0][0]
                reward = env.P[state][action][0][2]
                nextState = env.P[state][action][0][1]
                done = env.P[state][action][0][3]
                if done:
                    print("Terminating state")
                    actionRewardArray[action] = prob * (reward)
                else:
                    actionRewardArray[action] = prob * (reward + gamma * V[nextState])

            V[state] = np.max(actionRewardArray)
            delta = np.maximum(delta, np.abs(stateValue - V[state]))
            print("Delta {}".format(delta))
            print(actionRewardArray)
            policy[state] = np.argmax(actionRewardArray)
            print("Policy {} for state {}".format(policy[state], state))
            if state == 36:
                print(V)
                break

    return V, policy


V, policy = valueIteration(gamma=0.9, theta=0.001, iteration=1000, delta=1)

V_plot = V.reshape(4, 12)
policy_plot = policy.reshape(4, 12)
print(V_plot)
print(policy_plot)

fig = plt.figure(figsize=(4, 1))

fig.add_subplot(2, 1, 1)
plt.imshow(policy_plot)
plt.title("Value Iteration Policy")
plt.colorbar()

fig.add_subplot(2, 1, 2)
plt.imshow(V_plot)
plt.title("Value Iteration Value Function")
plt.colorbar()

plt.show()
print("Finished")