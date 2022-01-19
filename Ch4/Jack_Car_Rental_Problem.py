# https://towardsdatascience.com/elucidating-policy-iteration-in-reinforcement-learning-jacks-car-rental-problem-d41b34c8aec7

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import sys


class Poisson:
    def __init__(self, exp_num):
        self.exp_num = exp_num
        eps = 0.01
        
        # [alpha , beta] is the range of n's for which the pmf value is above eps
        self.alpha = 0
        state = 1
        self.vals = {}
        summer = 0
        
        while(1):
            if state == 1:
                temp = poisson.pmf(self.alpha, self.exp_num) 
                if(temp <= eps):
                    self.alpha+=1
                else:
                    self.vals[self.alpha] = temp
                    summer += temp
                    self.beta = self.alpha+1
                    state = 2
            elif state == 2:
                temp = poisson.pmf(self.beta, self.exp_num)
                if(temp > eps):
                    self.vals[self.beta] = temp
                    summer += temp
                    self.beta+=1
                else:
                    break    
        
        # normalizing the pmf, values of n outside of [alpha, beta] have pmf = 0
        added_val = (1-summer)/(self.beta-self.alpha)
        for key in self.vals:
            self.vals[key] += added_val
 
    def f(self, n):
        try:
            Ret_value = self.vals[n]
        except(KeyError):
            Ret_value = 0
        finally:
            return Ret_value

# A class holding the properties of a location together
class location:
    def __init__(self, req, ret):
        self.alpha = req  # value of lambda for requests
        self.beta = ret # value of lambda for returns
        self.poisson_alp = Poisson(self.alpha)
        self.poisson_beta = Poisson(self.beta)


class jcp:
    def __init__(self, max_cars, disc_rate, credit_reward, moving_reward):
        self.max_cars = max_cars
        self.disc_rate = disc_rate
        self.credit_reward = credit_reward
        self.moving_reward = moving_reward
        self.policy_evaluation_eps = 50
        self.save_policy_counter = 0
        self.save_value_counter = 0

        # Location initialisation
        self.A = location(3, 3)
        self.B = location(4, 2)

        # Initializing the value and policy matrices. Initial policy has zero value for all states.
        self.value = np.zeros((self.max_cars+1, self.max_cars+1))
        self.policy = np.zeros((self.max_cars+1, self.max_cars+1)).astype(int)

    def expected_reward(self, state, action):
        """
        state  : It's a pair of integers, # of cars at A and at B
        action : # of cars transferred from A to B,  -5 <= action <= 5 
        """
        reward = 0
        new_state = [max(min(state[0] - action, self.max_cars), 0) , max(min(state[1] + action, self.max_cars), 0)]
        
        # adding reward for moving cars from one location to another (which is negative)
        reward = reward + self.moving_reward * abs(action)
        
        #there are four discrete random variables which determine the probability distribution of the reward and next state
        for Aalpha in range(self.A.poisson_alp.alpha, self.A.poisson_alp.beta):
            for Balpha in range(self.B.poisson_alp.alpha, self.B.poisson_alp.beta):
                for Abeta in range(self.A.poisson_beta.alpha, self.A.poisson_beta.beta):
                    for Bbeta in range(self.B.poisson_beta.alpha, self.B.poisson_beta.beta):
                        """
                        Aalpha : sample of cars requested at location A
                        Abeta : sample of cars returned at location A
                        Balpha : sample of cars requested at location B
                        Bbeta : sample of cars returned at location B
                        prob_event  : probability of this event happening
                        """

                        # all four variables are independent of each other
                        prob_event = self.A.poisson_alp.vals[Aalpha] * self.B.poisson_alp.vals[Balpha] * \
                                        self.A.poisson_beta.vals[Abeta] * self.B.poisson_beta.vals[Bbeta]
                        
                        valid_requests_A = min(new_state[0], Aalpha)
                        valid_requests_B = min(new_state[1], Balpha)
                        
                        rew = (valid_requests_A + valid_requests_B)*(self.credit_reward)
                        
                        #calculating the new state based on the values of the four random variables
                        new_s = [0, 0]
                        new_s[0] = max(min(new_state[0] - valid_requests_A + Abeta, self.max_cars), 0)
                        new_s[1] = max(min(new_state[1] - valid_requests_B + Bbeta, self.max_cars), 0)
                        
                        #Bellman's equation
                        reward += prob_event * (rew + self.disc_rate * self.value[new_s[0]][new_s[1]])
                        
        return reward


    def policy_evaluation(self):
        # here policy_evaluation has a static variable eps whose values decreases over time
        eps = self.policy_evaluation_eps
        self.policy_evaluation_eps /= 10 
        
        while(1):
            delta = 0
            for i in range(self.value.shape[0]):
                for j in range(self.value.shape[1]):
                    # value[i][j] denotes the value of the state [i, j]
                    old_val = self.value[i][j]
                    self.value[i][j] = self.expected_reward([i, j], self.policy[i][j])
                    delta = max(delta, abs(self.value[i][j] - old_val))
                    print('.', end = '')
                    sys.stdout.flush()

            print(delta)
            sys.stdout.flush()
            if delta < eps:
                break

    def policy_improvement(self):
        policy_stable = True
        for i in range(self.value.shape[0]):
            for j in range(self.value.shape[1]):
                old_action = self.policy[i][j]
                
                max_act_val = None
                max_act = None
                
                move12 = min(i, 5) # if I have say 3 cars at the first location, then I can atmost move 3 from 1 to 2
                move21 = -min(j, 5) # if I have say 2 cars at the second location, then I can atmost move 2 from 2 to 1
                
                for act in range(move21, move12+1):
                    exp_reward = self.expected_reward([i, j], act)
                    if max_act_val == None:
                        max_act_val = exp_reward
                        max_act = act
                    elif max_act_val < exp_reward:
                        max_act_val = exp_reward
                        max_act = act
                    
                self.policy[i][j] = max_act
                
                if old_action != self.policy[i][j]:
                    policy_stable = False

        return policy_stable

    def run(self):
        while(1):
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            self.save_value()
            self.save_policy()
            if policy_stable == True:
                break
            
    def save_policy(self):
        self.save_policy_counter += 1
        ax = sns.heatmap(self.policy, linewidth=0.5)
        ax.invert_yaxis()
        plt.savefig('policy'+str(self.save_policy_counter)+'.svg')
        plt.close()
        
    def save_value(self):
        self.save_value_counter += 1
        ax = sns.heatmap(self.value, linewidth=0.5)
        ax.invert_yaxis()
        plt.savefig('value'+ str(self.save_value_counter)+'.svg')
        plt.close()


def main():
    jcp_obj = jcp(20, 0.9, 10, -2)
    jcp_obj.run()

if __name__ == '__main__':
    main()