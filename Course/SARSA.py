import gym
import numpy as np
import matplotlib.pyplot as plt
from Course.MovingAverage import MovingAverage

#UP    = 0
#RIGHT = 1
#DOWN  = 2
#LEFT  = 3

env = gym.make('CliffWalking-v0')


def Sarsa(nr_episodes, max_steps, gamma, alpha, epsilon_start, epsilon_end, nr_eps_steps, evaluate, Q_eval = None):
    eps_array = np.linspace(epsilon_start, epsilon_end, nr_eps_steps)
    finalReward = np.zeros((nr_episodes, nr_eps_steps))

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
                visitedStates[state_p] += 1
                if not done and step < max_steps:
                    # Update state action value
                    learned_value = reward + gamma * Q[state_p, action_p]
                    Q[state, action] = Q[state, action]+alpha*(learned_value-Q[state, action])
                    finalReward[it,eps_iter] += reward
                else:
                    finalReward[it,eps_iter] = finalReward[it, eps_iter]
                    break

                state=state_p
                step += 1

    Q_max = np.zeros((env.observation_space.n, 1))
    Q_plot = np.zeros((env.observation_space.n, 1))
    for state in range(env.observation_space.n):
        Q_plot[state] = np.max(Q[state, :])
        Q_max[state] = np.argmax(Q[state, :])
    print("Training for epsilon: %.2f"%(epsilon))
    print(Q_max.reshape(4, 12))
    print(Q_plot.reshape(4, 12))
    print(visitedStates.reshape(4, 12))

    plt.close('all')
    plt.imshow(Q_plot.reshape(4, 12))
    plt.colorbar()
    plt.title('CliffWalking SARSA' % (epsilon))
    legends = []
    movingAverageNum = 30

    plt.figure()
    for i in range(nr_eps_steps):
        plt.plot(finalReward[:, i], alpha=0.2)
        legends.append("epsilon=%.2f" % (eps_array[i]))
        plt.plot(MovingAverage(finalReward[:, i], movingAverageNum))
        legends.append("Moving Average epsilon=%.2f" % (eps_array[i]))

    plt.gca().legend((legends))
    plt.title('CliffWalking SARSA')
    plt.xlabel('Number of episodes')
    plt.ylabel('Reward per episode')
    axes = plt.gca()
    axes.set_xlim([-10,nr_episodes-movingAverageNum])
    axes.set_ylim([np.min(MovingAverage(finalReward[:,i],movingAverageNum)),10])
    plt.show()


Sarsa(nr_episodes=2000, max_steps=200, gamma=0.5, alpha=0.2, epsilon_start=0.2, epsilon_end=0.2, nr_eps_steps=1,
      evaluate=False)
    


