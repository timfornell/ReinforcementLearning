import gym
import numpy as np
import matplotlib.pyplot as plt
from Course import MovingAverage

#UP    = 0
#RIGHT = 1
#DOWN  = 2
#LEFT  = 3

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

def Sarsa(nr_episodes, max_steps, gamma, alpha, epsilon_start, epsilon_end, nr_eps_steps, evaluate, Q_eval = None,
          stochastic=False):
    eps_array = np.linspace(epsilon_start, epsilon_end, nr_eps_steps)
    finalReward = np.zeros((nr_episodes, nr_eps_steps))

    if stochastic:
        setTransitionProbabilites()
        
    for eps_iter in range(nr_eps_steps):
        epsilon = eps_array[eps_iter]

        Q = None
        if not evaluate:
            Q = np.random.uniform(0, 1, (env.observation_space.n, env.action_space.n))
        else:
            Q = Q_eval

        visitedStates = np.zeros((env.observation_space.n, 1))

        for it in range(nr_episodes):

            # Initialize state
            state = env.reset()
            step = 0
            while step<max_steps:
                #env.render()
            # greedy action selection
                if np.random.uniform(low=0, high=1) < (1 - epsilon):
                    action = np.argmax(Q[state, :])
                else:
                    action = env.action_space.sample()
                state_p, reward, done, info = env.step(action) # take action according to policy
            # policy update
                action_p = np.argmax(Q[state_p, :])
                visitedStates[state] += 1
                if not done and step < max_steps:
                    # Update state action value
                    learned_value = reward + gamma * Q[state_p, action_p]
                    Q[state, action] = Q[state, action]+alpha*(learned_value-Q[state, action])
                    finalReward[it,eps_iter] += reward
                else:
                    visitedStates[state_p] += 1
                    finalReward[it,eps_iter] = finalReward[it, eps_iter]
                    break

                state=state_p
                step += 1

    return Q, visitedStates, finalReward

Q, visitedStates, finalReward=Sarsa(nr_episodes=2000, max_steps=100, gamma=0.9, alpha=0.8, epsilon_start=0.1, epsilon_end=0.1, nr_eps_steps=1,evaluate=False)
Q_eval, visitedStates_eval, finalReward_eval=Sarsa(nr_episodes=1, max_steps=100, gamma=0.9, alpha=0.8, epsilon_start=0, epsilon_end=0, nr_eps_steps=1,evaluate=True, Q_eval=Q)
Q_stoch, visitedStates_stoch, finalReward_stoch=Sarsa(nr_episodes=2000, max_steps=100, gamma=0.9, alpha=0.8, epsilon_start=0.1, epsilon_end=0.1, nr_eps_steps=1,evaluate=False,stochastic=True)

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
print(visitedStates_eval.reshape(4, 12))

fig1 = plt.figure(figsize=(6, 1))

fig1.add_subplot(3, 1, 1)
plt.imshow(Q_plot.reshape(4, 12))
plt.title("SARSA, State Action function")
plt.colorbar()

fig1.add_subplot(3, 1, 2)
plt.imshow(Q_max.reshape(4, 12))
plt.title("SARSA, Policy")
plt.colorbar()

fig1.add_subplot(3, 1, 3)
plt.imshow(visitedStates.reshape(4, 12))
plt.title("SARSA, number of visits per state")
plt.colorbar()

fig2 = plt.figure(figsize=(6, 1))
fig2.add_subplot(3, 1, 1)
plt.imshow(Q_plot_stoch.reshape(4, 12))
plt.title("SARSA, State Action function Stochastic")
plt.colorbar()

fig2.add_subplot(3, 1, 2)
plt.imshow(Q_max_stoch.reshape(4, 12))
plt.title("SARSA, Policy Stochastic")
plt.colorbar()

fig2.add_subplot(3, 1, 3)
plt.imshow(visitedStates_stoch.reshape(4, 12))
plt.title("SARSA, number of visits per state Stochastic")
plt.colorbar()

plt.figure(3)
plt.imshow(visitedStates_eval.reshape(4, 12))
plt.title("SARSA, Visited states during evaluation")
plt.colorbar()

plt.figure(4)
plt.title("SARSA, total expected reward")
plt.clf()
finalReward_aver = MovingAverage.MovingAverage(finalReward, 10)
plt.plot(finalReward, alpha=0.5)
plt.plot(finalReward_aver, alpha=0.5)
plt.title("SARSA")
plt.legend(["Reward", "Smoothed reward"])
plt.xlabel("Iterations")
plt.ylabel("Reward")

plt.figure(5)
plt.title("SARSA, total expected reward stochastic")
plt.clf()
finalReward_aver = MovingAverage.MovingAverage(finalReward_stoch, 10)
plt.plot(finalReward_stoch, alpha=0.5)
plt.plot(finalReward_aver, alpha=0.5)
plt.title("SARSA")
plt.legend(["Reward", "Smoothed reward"])
plt.xlabel("Iterations")
plt.ylabel("Reward")

plt.show()


