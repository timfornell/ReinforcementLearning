import gym
import numpy as np
import matplotlib.pyplot as plt

def MovingAverage(a, n=3):
    best = np.array(a)
    ret = np.zeros((best.size-n+1,1))
    for i in range(best.size-n+1):
        ret[i] = np.sum(best[i:i+n])/n
    return ret

def setTransitionProbabilites():
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            items = list(env.P[state][action][0])
            if action == 0:
                items[0] = 0.85
            else:
                items[0] = 0.05
            env.P[state][action][0] = tuple(items)

#UP    = 0
#RIGHT = 1
#DOWN  = 2
#LEFT  = 3

env = gym.make('CliffWalking-v0')

def ExpectedSarsa(nr_episodes, max_steps, gamma, alpha, epsilon, evaluate, Q_eval = None, stochastic=False):
    Q = None
    if not evaluate:
        Q = np.random.uniform(0, 1, (env.observation_space.n, env.action_space.n))
    else:
        Q = Q_eval

    if stochastic:
        setTransitionProbabilites()

    pi= np.zeros((env.action_space.n,env.observation_space.n))
    visitedStates = np.zeros((env.observation_space.n, 1))
    finalReward = np.zeros((nr_episodes, 1))

    for it in range(nr_episodes):

        # Initialize state
        state = env.reset()
        step = 0
        visitedStates[state] += 1

        while step<max_steps:
            #env.render()

            # greedy action selection
            if np.random.uniform(low=0, high=1) < (1 - epsilon):
                action = np.argmax(Q[state, :])
            else:
                action = env.action_space.sample()

            state_p, reward, done, info = env.step(action) # take action according to policy

            # policy update
            pi[:,state_p]=epsilon/(env.action_space.n)
            action_p = np.argmax(Q[state_p, :])
            pi[action_p,state_p] =1-epsilon+epsilon/env.action_space.n

            visitedStates[state_p] += 1

            if not done and step < max_steps:
                # Update state action value
                learned_value = reward + gamma * np.dot(pi[:,state_p],Q[state_p, :])
                Q[state, action] = Q[state, action]+alpha*(learned_value-Q[state, action])
                finalReward[it] += reward
            else:
                finalReward[it] = finalReward[it] / step
                break

            state=state_p
            step += 1

    return Q, visitedStates, finalReward


Q, visitedStates, finalReward = ExpectedSarsa(nr_episodes=2000, max_steps=100, gamma=0.7, alpha=0.8, epsilon=0.3,
                                              evaluate=False)
Q_eval, visitedStates_eval, finalReward_eval = ExpectedSarsa(nr_episodes=1, max_steps=100, gamma=0.7, alpha=0.8,
                                                             epsilon=0, evaluate=True, Q_eval=Q)
Q_stoch, visitedStates_stoch, finalReward_stoch = ExpectedSarsa(nr_episodes=2000, max_steps=100, gamma=0.7, alpha=0.8,
                                                            epsilon=0.3, evaluate=False, stochastic=True)

Q_max = np.zeros((env.observation_space.n, 1))
Q_plot = np.zeros((env.observation_space.n, 1))
for state in range(48):
    Q_plot[state] = np.max(Q[state, :])
    Q_max[state] = np.argmax(Q[state, :])

print(Q_max.reshape(4, 12))
print(Q_plot.reshape(4, 12))
print(visitedStates.reshape(4, 12))
fig1 = plt.figure(figsize=(6, 1))

fig1.add_subplot(3, 1, 1)
plt.imshow(Q_plot.reshape(4, 12))
plt.title("Expected Sarsa, State Action function")
plt.colorbar()

fig1.add_subplot(3, 1, 2)
plt.imshow(Q_max.reshape(4, 12))
plt.title("Expected Sarsa, Policy")
plt.colorbar()

fig1.add_subplot(3, 1, 3)
plt.imshow(visitedStates.reshape(4, 12))
plt.title("Expected Sarsa, number of visits per state")
plt.colorbar()

plt.figure(2)
plt.imshow(visitedStates_eval.reshape(4, 12))
plt.title("Expected Sarsa, Visited states during evaluation")
plt.colorbar()

plt.figure()
plt.plot(finalReward)
plt.plot(MovingAverage(finalReward,10))
plt.legend(('Reward per episode','Average reward per episode'),loc='lower right')
plt.title('CliffWalking ExpectedSARSA')
plt.xlabel('Number of episodes')
plt.ylabel('Reward per episode')
plt.show()
