import gym
import numpy as np
import matplotlib.pyplot as plt
from Course import MovingAverage

env = gym.make('CliffWalking-v0')


def setTransitionProbabilites():
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            items = list(env.P[state][action][0])
            if action == 0:
                items[0] = 0.85
            else:
                items[0] = 0.05
            env.P[state][action][0] = tuple(items)


def QLearning(iterations, max_steps, gamma, alpha, epsilon, evaluate, Q_eval=None, stochastic=False):
    # Q = np.zeros((env.observation_space.n, env.action_space.n))
    Q = None
    if not evaluate:
        Q = np.random.uniform(0, 1, (env.observation_space.n, env.action_space.n))
    else:
        Q = Q_eval

    if stochastic:
        setTransitionProbabilites()

    visitedStates = np.zeros((env.observation_space.n, 1))

    finalReward = np.zeros((iterations, 1))

    for it in range(iterations):
        current_state = env.reset()

        step = 0
        while step < max_steps:
            #env.render()

            action = 0
            if np.random.uniform(low=0, high=1) < 1 - epsilon:
                action = np.argmax(Q[current_state, :])
            else:
                action = env.action_space.sample()

            state2, reward, done, info = env.step(action) # take a random action

            old_state = current_state
            current_state = state2
            visitedStates[old_state] += 1

            if not done and step < max_steps:
                finalReward[it] += reward
                learned_value = reward + gamma * np.max(Q[current_state, :])
                old_value = Q[old_state, action]
                Q[old_state, action] = (1 - alpha) * old_value + alpha * learned_value
            else:
                visitedStates[current_state] += 1
                break

            step += 1
            if stochastic:
                epsilon = epsilon / (it + 1)

    return Q, visitedStates, finalReward


Q, visitedStates, finalReward = QLearning(iterations=1000, max_steps=100, gamma=0.5, alpha=0.2, epsilon=0.5,
                                          evaluate=False)
Q_eval, visitedStates_eval, finalReward_eval = QLearning(iterations=1, max_steps=100, gamma=0.5, alpha=0.2, epsilon=0,
                                                         evaluate=True, Q_eval=Q)

Q_stoch, visitedStates_stoch, finalReward_stoch = QLearning(iterations=1000, max_steps=100, gamma=0.5, alpha=0.2,
                                                            epsilon=0.5, evaluate=False, stochastic=True)

Q_max = np.zeros((env.observation_space.n, 1))
Q_plot = np.zeros((env.observation_space.n, 1))
Q_max_stoch = np.zeros((env.observation_space.n, 1))
Q_plot_stoch = np.zeros((env.observation_space.n, 1))
for state in range(48):
    Q_plot[state] = np.max(Q[state, :])
    Q_max[state] = np.argmax(Q[state, :])
    Q_plot_stoch[state] = np.max(Q_stoch[state, :])
    Q_max_stoch[state] = np.argmax(Q_stoch[state, :])

print(Q_max.reshape(4, 12))
print(Q_plot.reshape(4, 12))

print(visitedStates.reshape(4, 12))

fig1 = plt.figure(figsize=(6, 1))

fig1.add_subplot(3, 1, 1)
plt.imshow(Q_plot.reshape(4, 12))
plt.title("Q-Learning, State Action function")
plt.colorbar()

fig1.add_subplot(3, 1, 2)
plt.imshow(Q_max.reshape(4, 12))
plt.title("Q-Learning, Policy")
plt.colorbar()

fig1.add_subplot(3, 1, 3)
plt.imshow(visitedStates.reshape(4, 12))
plt.title("Q-Learning, number of visits per state")
plt.colorbar()

fig2 = plt.figure(figsize=(6, 1))
fig2.add_subplot(3, 1, 1)
plt.imshow(Q_plot_stoch.reshape(4, 12))
plt.title("Q-Learning, State Action function Stochastic")
plt.colorbar()

fig2.add_subplot(3, 1, 2)
plt.imshow(Q_max_stoch.reshape(4, 12))
plt.title("Q-Learning, Policy Stochastic")
plt.colorbar()

fig2.add_subplot(3, 1, 3)
plt.imshow(visitedStates_stoch.reshape(4, 12))
plt.title("Q-Learning, number of visits per state Stochastic")
plt.colorbar()

plt.figure(3)
plt.imshow(visitedStates_eval.reshape(4, 12))
plt.title("Q-Learning, Visited states during evaluation")
plt.colorbar()

plt.figure(4)
plt.title("Q-Learning, total expected reward")
plt.clf()
finalReward_aver = MovingAverage.MovingAverage(finalReward, 10)
plt.plot(finalReward, alpha=0.5)
plt.plot(finalReward_aver, alpha=0.5)
plt.title("Q-Learning")
plt.legend(["Reward", "Smoothed reward"])
plt.xlabel("Iterations")
plt.ylabel("Reward")

plt.figure(5)
plt.title("Q-Learning, total expected reward")
plt.clf()
finalReward_aver = MovingAverage.MovingAverage(finalReward_stoch, 10)
plt.plot(finalReward_stoch, alpha=0.5)
plt.plot(finalReward_aver, alpha=0.5)
plt.title("Q-Learning")
plt.legend(["Reward", "Smoothed reward"])
plt.xlabel("Iterations")
plt.ylabel("Reward")

plt.show()
