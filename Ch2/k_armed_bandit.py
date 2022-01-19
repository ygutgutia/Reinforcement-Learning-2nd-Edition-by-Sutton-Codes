# https://www.datahubbs.com/multi_armed_bandits_reinforcement_learning_1/
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

class eps_bandit:
    '''
    epsilon-greedy k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    eps: probability of random action 0 < eps < 1 (float)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, eps, iters, mu='random'):
        self.k = k
        self.eps = eps
        self.iters = iters
        self.n = 0 # Step count
        self.k_n = np.zeros(k) # Step count for each arm
        self.mean_reward = 0 # Total mean reward
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(k) # Mean reward for each arm
        
        if type(mu) == list or type(mu).__module__ == np.__name__:        
            self.mu = np.array(mu)
        elif mu == 'random':
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            self.mu = np.linspace(0, k-1, k)
        
    def pull(self):
        p = np.random.rand()
        if self.eps == 0 and self.n == 0:
            a = np.random.choice(self.k)
        elif p < self.eps:
            a = np.random.choice(self.k)
        else:
            a = np.argmax(self.k_reward)
        reward = np.random.normal(self.mu[a], 1)
        
        self.n += 1
        self.k_n[a] += 1
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward
            
    def reset(self):
        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(k)

def main(k, iters, episodes):
    eps_0_rewards = np.zeros(iters)
    eps_001_rewards = np.zeros(iters)
    eps_01_rewards = np.zeros(iters)
    eps_0_selection = np.zeros(k)
    eps_001_selection = np.zeros(k)
    eps_01_selection = np.zeros(k)

    # Run experiments
    for i in range(episodes):
        # Initialize bandits
        eps_0 = eps_bandit(k, 0, iters)
        eps_001 = eps_bandit(k, 0.01, iters, eps_0.mu.copy())
        eps_01 = eps_bandit(k, 0.1, iters, eps_0.mu.copy())
        
        # Run experiments
        eps_0.run()
        eps_001.run()
        eps_01.run()
        
        # Update long-term averages
        eps_0_rewards = eps_0_rewards + (eps_0.reward - eps_0_rewards) / (i + 1)
        eps_001_rewards = eps_001_rewards + (eps_001.reward - eps_001_rewards) / (i + 1)
        eps_01_rewards = eps_01_rewards + (eps_01.reward - eps_01_rewards) / (i + 1)
        
        # Average actions per episode
        eps_0_selection = eps_0_selection + (eps_0.k_n - eps_0_selection) / (i + 1)
        eps_001_selection = eps_001_selection + (eps_001.k_n - eps_001_selection) / (i + 1)
        eps_01_selection = eps_01_selection + (eps_01.k_n - eps_01_selection) / (i + 1)
        print(i)
        
    plt.figure(figsize=(12, 8))
    plt.plot(eps_0_rewards, label="epsilon=0 (greedy)")
    plt.plot(eps_001_rewards, label="epsilon=0.01")
    plt.plot(eps_01_rewards, label="epsilon=0.1")
    for i in range(k):
        plt.hlines(eps_0.mu[i], xmin=0, xmax=iters, alpha=0.5, linestyle="--")
    plt.legend(bbox_to_anchor=(1.3, 0.5))
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average epsilon-greedy Rewards after " + str(episodes) + " Episodes")
    plt.legend()
    plt.show()

    bins = np.linspace(0, k-1, k)
    plt.figure(figsize=(12, 8))
    plt.bar(bins, eps_0_selection, width = 0.33, color='b', label="epsilon=0")
    plt.bar(bins+0.33, eps_001_selection, width=0.33, color='g', label="epsilon=0.01")
    plt.bar(bins+0.66, eps_01_selection,  width=0.33, color='r',label="epsilon=0.1")
    plt.legend(bbox_to_anchor=(1.2, 0.5))
    plt.xlim([0, k])
    plt.title("Actions Selected by Each Algorithm")
    plt.xlabel("Action")
    plt.ylabel("Number of Actions Taken")
    plt.legend()
    plt.show()

    opt_per = np.array([eps_0_selection, eps_001_selection, eps_01_selection]) / iters * 100
    df = pd.DataFrame(opt_per, index=['epsilon=0', 'epsilon=0.01', 'epsilon=0.1'], 
                columns=["a = " + str(x) for x in range(0, k)])
    print("Percentage of actions selected:")
    print(df)

if __name__ == '__main__':
    main(10, 2000, 50)