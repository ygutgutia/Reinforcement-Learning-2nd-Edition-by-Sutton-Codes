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
    
    def random_walk(self):
        mu_shift = np.random.normal(0, 0.01, self.k)
        self.mu += mu_shift

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
            self.random_walk()
            self.pull()
            self.reward[i] = self.mean_reward
            
    def reset(self):
        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(k)

def main(k, iters, episodes, step_size):
    eps_01_rewards = np.zeros(iters)
    eps_01_selection = np.zeros(k)
    eps_01_alpha_rewards = np.zeros(iters)
    eps_01_alpha_selection = np.zeros(k)

    # Run experiments
    for i in range(episodes):
        # Initialize bandit and Run experiment
        bandit_curr = eps_bandit(k, 0, iters)
        bandit_curr.run()
        
        # Update long-term averages
        eps_01_rewards = eps_01_rewards + (bandit_curr.reward - eps_01_rewards) / (i + 1)
        eps_01_alpha_rewards = eps_01_alpha_rewards + (bandit_curr.reward - eps_01_alpha_rewards) * step_size
        
        # Average actions per episode
        eps_01_selection = eps_01_selection + (bandit_curr.k_n - eps_01_selection) / (i + 1)
        eps_01_alpha_selection = eps_01_alpha_selection + (bandit_curr.k_n - eps_01_alpha_selection) * step_size
        print(i)
        
    plt.figure(figsize=(12, 8))
    plt.plot(eps_01_rewards, label="mean")
    plt.plot(eps_01_alpha_rewards, label="step_size")
    plt.legend(bbox_to_anchor=(1.3, 0.5))
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average epsilon-greedy Rewards after " + str(episodes) + " Episodes")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main(10, 2000, 1000, 0.1)
    main(10, 10000, 1000, 0.1)