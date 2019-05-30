    # -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:42:46 2019

@author: Markus
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
    import collections

episodes = 200
max_steps = 100
gamma = 0.8
epsilon = 0.1

debugStateVar = 33
debugActionVar = 1
debugEndState = 35

debugOptStateActions = [[32, 1],[33, 1],[34,1]]

print(debugStateVar, debugActionVar)


def isDebugVar(state, action = -1):
    if action == -1:
        return debugStateVar == state
    
    return (debugStateVar == state and debugActionVar == action)


def isInsideOptStates(state, action):
    if([state, action] in debugOptStateActions):
        return True
    return False


class MonteCarloControl():
        
    def __init__(self, env, episodes, gamma, epsilon, render):
        self.Q = np.random.uniform(-1, -1, (env.observation_space.n, env.action_space.n))
        self.policy = np.random.uniform(1/env.action_space.n, 1/env.action_space.n, (env.observation_space.n, env.action_space.n))
        self.gamma = gamma
        self.episodes = episodes
        self.Returns = collections.defaultdict(lambda : list())
        self.env = env
        self.epsilon = epsilon
        self.renderVar = render
        self.numberOfRenderedTimes = 0
        self.visitedStates =  np.zeros((env.observation_space.n, env.action_space.n))
        for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                self.Returns[state,action].append(-100)

    def Train(self):
        min_policy_return = collections.defaultdict(lambda : float())
        for ep in range(self.episodes):
            print(ep)
            episodeHistory, visitedStatesAndActions, returns, nrSteps = self.run_episode(False, 5000)
            self.visitedStates +=visitedStatesAndActions
            self.update_policy(episodeHistory, visitedStatesAndActions, returns, nrSteps)
            min_policy_return[ep] = self.calculate_min_policy_return(100)
            if self.renderVar and ep%np.ceil(self.episodes/10)==0:
                self.render()
                self.debugPrints()
        return min_policy_return
        
    def run_episode(self, followPolicy, maxSteps):     
        Rewards = collections.defaultdict(lambda : float())
        next_state = self.env.reset()
        current_state = next_state
        episodeHistory = []
        done = False
        visitedStatesAndActions = np.zeros((env.observation_space.n, env.action_space.n))
        nrSteps = 0
        while not done and not nrSteps > maxSteps:
            #self.env.render()
            
            if(followPolicy):
                next_action = np.argmax(self.policy[current_state])
            else:
                next_action = np.random.choice(self.env.action_space.n, p=self.policy[current_state])
            
            #if isInsideOptStates(current_state, next_action):
                #print("good action, state: ", current_state)
            visitedStatesAndActions[current_state, next_action] += 1;
            next_state, reward, done, info = self.env.step(next_action)
            Rewards[current_state,next_action] = reward
            episodeHistory.append([current_state, next_action, reward])
            current_state = next_state
            nrSteps +=1

        returns = collections.defaultdict(lambda : -100.0)
        returns[nrSteps-1] = episodeHistory[nrSteps-1][2]
        for i in reversed(range(nrSteps-1)):
            returns[i] = episodeHistory[i][2] + self.gamma*returns[i+1]
        return episodeHistory, visitedStatesAndActions, returns, nrSteps
    
    def update_policy(self, episodeHistory, visitedStatesAndActions, returns, nrSteps):
        #print(nrSteps)
        tempBool = False
        if True:
            doNothing = 0
        for i in reversed(range(nrSteps)):
            state, action = episodeHistory[i][0:2]
            #if (state,action) not in [tuple(x[0:2]) for x in episodeHistory[i+1:]]:
            tempBool = False
            
            for x in episodeHistory[i+1:]:
                if state == x[0]:
                    if action == x[1]:
                        tempBool = True
            if not tempBool:
                Gt = returns[i]
                self.Returns[state,action].append(Gt)
                
                self.Q[state,action] = np.average(self.Returns[state,action])
        for state in range(env.observation_space.n):
            A_star = np.argmax(self.Q[state,:])
            for action in range(env.action_space.n):
                if A_star == action:
                    self.policy[state,action] = 1 - self.epsilon + self.epsilon/self.env.action_space.n
                else:
                    self.policy[state, action] = self.epsilon/self.env.action_space.n

    def debugPrints(self):
        policy_render = np.argmax(self.policy, axis=1)
        print(policy_render.reshape(4,12))
        for state in range(debugStateVar, debugEndState+1):
            print("state: ", state)
            print(self.Q[state])

    def calculate_min_policy_return(self, maxSteps):
        episodeHistory, visitedStatesAndActions, returns, nrSteps = self.run_episode(True, maxSteps) # 0.0 since greedy actions should be choosed
        return sumFloatDictionary(returns)
                            
    def render(self):
        plt.figure(1)
        plt.close
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.imshow(self.Q.reshape(4,48))
        #if self.numberOfRenderedTimes == 0:
            #plt.colorbar()
        plt.subplot(2,1,2)
        plt.imshow(self.policy.reshape(4,48))
        #if self.numberOfRenderedTimes == 0:
            #plt.colorbar()
        
        self.numberOfRenderedTimes += 1
        plt.show()


def sumFloatDictionary(dic):
    sumVar = 0
    for i in range(len(dic)-1):
        if i==0:
            doNothing = 0
        if(type(dic[i]) == list):
            continue
        sumVar += dic[i]
    return sumVar


if __name__=="__main__":
    env = gym.make('CliffWalking-v0')
    agent = MonteCarloControl(env, episodes,gamma,epsilon, True)
    minPolicyReturn = agent.Train()
    plt.figure(2)
    plt.close
    plt.figure(2)
    plt.plot(minPolicyReturn.values())
    plt.figure(3)
    plt.close
    #plt.figure(3)
    print(np.nonzero(agent.visitedStates < 1)[0])
    #plt.colorbar()
    
                    