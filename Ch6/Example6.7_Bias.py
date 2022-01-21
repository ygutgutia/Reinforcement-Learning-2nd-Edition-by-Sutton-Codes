import numpy as np
import random
from matplotlib import pyplot as plt

 # TODO: Double QL is not improving over time although graph shape is as expected
class MaxBias_QL:
    def __init__(self, disc_factor, expl_cons, step_size, iters, b_states):
        self.iters = iters
        self.left_from_A = []
        self.step_size = step_size
        self.expl_cons = expl_cons
        self.b_states = b_states
        self.disc_factor = disc_factor
        self.action_value_A = np.zeros(2)
        self.action_value_B = np.zeros(b_states)

    def ql(self):
        self.left_count_from_A = 0
        for i in range(self.iters):
            self.run(i+1)
            print("Iteration Done : " + str(i))
    
    def find_optimal_action(self, currpos):
        curr_actions = self.action_value_A if currpos == 'A' else self.action_value_B
        max_actions = max(curr_actions)
        action = random.choice([i for i in range(len(curr_actions)) if curr_actions[i] == max_actions])
        return action

    def find_action(self, currpos):        
        exploration_prob = np.random.uniform()
        action = np.random.choice(2 if currpos == 'A' else self.b_states)
        if(exploration_prob >= self.expl_cons):
            action = self.find_optimal_action(currpos)
        return action

    def get_new_state(self, currpos, action):
        if currpos == 'A':
            if action == 0:
                return 0, 'B'
            return 0, 'C'
        reward = np.random.normal(-0.1, 1)
        return reward, 'C'

    def expected_award(self, currpos):
        if(currpos == 'C'):
            return 'C'
        action = self.find_action(currpos)
        reward, newpos = self.get_new_state(currpos, action)
        new_action = self.find_optimal_action(newpos)
        next_qsa = self.action_value_A[new_action] if newpos == 'A' else (self.action_value_B[new_action] if newpos == 'B' else 0)

        if currpos == 'A':
            self.action_value_A[action] += self.step_size * (reward + next_qsa * self.disc_factor - self.action_value_A[action])
        else:
            self.action_value_B[action] += self.step_size * (reward + next_qsa * self.disc_factor - self.action_value_B[action])
        return newpos

    def run(self, curr_iter):
        currpos = 'A'
        while not(currpos == 'C'):
            newpos = self.expected_award(currpos)
            if(currpos == 'A' and newpos == 'B'):
                self.left_count_from_A += 1
            currpos = newpos
        self.left_from_A.append(self.left_count_from_A * 100 / curr_iter)


class MaxBias_DoubleQL:
    def __init__(self, disc_factor, expl_cons, step_size, iters, b_states):
        self.iters = iters
        self.left_from_A = []
        self.step_size = step_size
        self.expl_cons = expl_cons
        self.b_states = b_states
        self.disc_factor = disc_factor
        self.action_value_A1 = np.zeros(2)
        self.action_value_A2 = np.zeros(2)
        self.action_value_B1 = np.zeros(b_states)
        self.action_value_B2 = np.zeros(b_states)

    def qql(self):
        self.left_count_from_A = 0
        for i in range(self.iters):
            self.run(i+1)
            print("Iteration Done : " + str(i))
    
    def find_optimal_action(self, curr_actions):
        max_actions = max(curr_actions)
        action = random.choice([i for i in range(len(curr_actions)) if curr_actions[i] == max_actions])
        return action

    def find_action(self, currpos):        
        exploration_prob = np.random.uniform()
        action = np.random.choice(2 if currpos == 'A' else self.b_states)
        if(exploration_prob >= self.expl_cons):
            if currpos == 'A':
                action = self.find_optimal_action(self.action_value_A1 + self.action_value_A2)
            else:
                action = self.find_optimal_action(self.action_value_B1 + self.action_value_B2)
        return action

    def get_new_state(self, currpos, action):
        if currpos == 'A':
            if action == 0:
                return 0, 'B'
            return 0, 'C'
        reward = np.random.normal(-0.1, 1)
        return reward, 'C'

    def expected_award(self, currpos):
        if(currpos == 'C'):
            return 'C'
        action = self.find_action(currpos)
        reward, newpos = self.get_new_state(currpos, action)

        update_prob = np.random.uniform()
        if currpos == 'A':
            if update_prob < 0.5:
                new_action = self.find_optimal_action(self.action_value_A1)
                next_qsa = self.action_value_A2[new_action]
                self.action_value_A1[action] += self.step_size * (reward + next_qsa * self.disc_factor - self.action_value_A1[action])
            else:
                new_action = self.find_optimal_action(self.action_value_A2)
                next_qsa = self.action_value_A1[new_action]
                self.action_value_A2[action] += self.step_size * (reward + next_qsa * self.disc_factor - self.action_value_A2[action])
        else:
            if update_prob < 0.5:
                new_action = self.find_optimal_action(self.action_value_B1)
                next_qsa = self.action_value_B1[new_action]
                self.action_value_B1[action] += self.step_size * (reward + next_qsa * self.disc_factor - self.action_value_B1[action])
            else:
                new_action = self.find_optimal_action(self.action_value_B2)
                next_qsa = self.action_value_B2[new_action]
                self.action_value_B2[action] += self.step_size * (reward + next_qsa * self.disc_factor - self.action_value_B2[action])
        return newpos

    def run(self, curr_iter):
        currpos = 'A'
        while not(currpos == 'C'):
            newpos = self.expected_award(currpos)
            if(currpos == 'A' and newpos == 'B'):
                self.left_count_from_A += 1
            currpos = newpos
        self.left_from_A.append(self.left_count_from_A * 100 / curr_iter)


def plot(qlobj, qqlobj):
    plt.plot(qlobj, label='QLearning')
    plt.plot(qqlobj, label='Double QLearning')
    plt.legend()
    plt.show()

def main():
    objq = MaxBias_QL(1, 0.1, 0.1, 1000, 5)
    objq.ql()
    objqq = MaxBias_DoubleQL(1, 0.1, 0.1, 1000, 5)
    objqq.qql()
    plot(objq.left_from_A, objqq.left_from_A)


if __name__ == '__main__':
    main()