import numpy as np
import random
from matplotlib import pyplot as plt


# TODO: Q Learning performing better than Sarsa
# 0 - L, 1 - U, 2 - R, 3 - D
action_map = [[-1, 0], [0, 1], [1, 0], [0, -1]]

class CliffWalking_sarsa:
    def __init__(self, maxx, maxy, cliff_height, disc_factor, expl_cons, step_size, iters):
        self.maxx = maxx
        self.maxy = maxy
        self.iters = iters
        self.possible_actions = len(action_map)
        self.step_size = step_size
        self.expl_cons = expl_cons
        self.disc_factor = disc_factor
        self.start_state = [0, 0]
        self.end_state = [maxx-1, 0]
        self.episode_reward = []
        self.cliff_height = cliff_height # maxy size array
        self.action_value = np.zeros(shape=(maxx, maxy, self.possible_actions))

    def sarsa(self):
        self.curr_total_reward = 0
        self.iter_curr = 0
        for i in range(self.iters):
            self.run()
            print("Iteration Done : " + str(i) + " Reward Gained = "+ str(self.curr_total_reward))
    
    def find_optimal_action(self, currx, curry):
        curr_actions = self.action_value[currx][curry]
        max_actions = max(curr_actions)
        action = random.choice([i for i in range(len(curr_actions)) if curr_actions[i] == max_actions])
        return action

    def find_action(self, currx, curry):        
        exploration_prob = np.random.uniform()
        action = action = np.random.choice(self.possible_actions)
        if(exploration_prob >= self.expl_cons):
            action = self.find_optimal_action(currx, curry)
        return action

    def get_new_state(self, currx, curry, action):
        newx = currx + action_map[action][0]
        newy = curry + action_map[action][1]
        newx = min(self.maxx-1, max(0, newx))
        newy = min(self.maxy-1, max(0, newy))
        reward = -1
        if([currx, curry] == self.end_state):
            reward = 0
        if(newy <= self.cliff_height and not(newx == 0 or newx == self.maxx-1)):
            newx = 0
            newy = 0
            reward = -100
        return newx, newy, reward

    def expected_award(self, currx, curry, action):
        if(currx == self.end_state[0] and curry == self.end_state[1]):
            return -1, -1, -1
        if(currx < 0 or curry < 0 or currx >= self.maxx or curry >= self.maxy):
            return -1, -1, -1

        newx, newy, reward = self.get_new_state(currx, curry, action)
        new_action = self.find_action(newx, newy)
        next_qsa = self.action_value[newx][newy][new_action]
        self.action_value[currx][curry][action] += self.step_size * (reward + \
                            next_qsa * self.disc_factor - self.action_value[currx][curry][action])

        self.curr_total_reward += reward
        return newx, newy, new_action

    def run(self):
        self.iter_curr += 1
        curr_pos = self.start_state
        currx, curry, action = curr_pos[0], curr_pos[1], self.find_action(curr_pos[0], curr_pos[1])
        while not(currx == -1 and curry == -1 and action == -1):
            currx, curry, action = self.expected_award(currx, curry, action)
        self.episode_reward.append(self.curr_total_reward/self.iter_curr)
    
    def print_policy(self):
        print("State Action Value are : ")
        print(self.action_value)
        print("Optimal Policy is : ")
        for i in range(self.maxx):
            for j in range(self.maxy):
                a = self.find_optimal_action(i, j)
                letter = 'L' if a == 0 else ('U' if a == 1 else ('R' if a == 2 else 'D'))
                print(letter, end = '')
            print()
        
class CliffWalking_ql:
    def __init__(self, maxx, maxy, cliff_height, disc_factor, expl_cons, step_size, iters):
        self.maxx = maxx
        self.maxy = maxy
        self.iters = iters
        self.possible_actions = len(action_map)
        self.step_size = step_size
        self.expl_cons = expl_cons
        self.disc_factor = disc_factor
        self.start_state = [0, 0]
        self.end_state = [maxx-1, 0]
        self.episode_reward = []
        self.cliff_height = cliff_height # maxy size array
        self.action_value = np.zeros(shape=(maxx, maxy, self.possible_actions))

    def ql(self):
        self.curr_total_reward = 0
        self.iter_curr = 0
        for i in range(self.iters):
            self.run()
            print("Iteration Done : " + str(i) + " Reward Gained = "+ str(self.curr_total_reward))
    
    def find_optimal_action(self, currx, curry):
        curr_actions = self.action_value[currx][curry]
        max_actions = max(curr_actions)
        action = random.choice([i for i in range(len(curr_actions)) if curr_actions[i] == max_actions])
        return action

    def find_action(self, currx, curry):        
        exploration_prob = np.random.uniform()
        action = action = np.random.choice(self.possible_actions)
        if(exploration_prob >= self.expl_cons):
            action = self.find_optimal_action(currx, curry)
        return action

    def get_new_state(self, currx, curry, action):
        newx = currx + action_map[action][0]
        newy = curry + action_map[action][1]
        newx = min(self.maxx-1, max(0, newx))
        newy = min(self.maxy-1, max(0, newy))
        reward = -1
        if([currx, curry] == self.end_state):
            reward = 0
        if(newy <= self.cliff_height and not(newx == 0 or newx == self.maxx-1)):
            newx = 0
            newy = 0
            reward = -100
        return newx, newy, reward

    def expected_award(self, currx, curry):
        if(currx == self.end_state[0] and curry == self.end_state[1]):
            return -1, -1
        if(currx < 0 or curry < 0 or currx >= self.maxx or curry >= self.maxy):
            return -1, -1

        action = self.find_action(currx, curry)
        newx, newy, reward = self.get_new_state(currx, curry, action)
        new_action = self.find_optimal_action(newx, newy)
        next_qsa = self.action_value[newx][newy][new_action]
        self.action_value[currx][curry][action] += self.step_size * (reward + \
                            next_qsa * self.disc_factor - self.action_value[currx][curry][action])

        self.curr_total_reward += reward
        return newx, newy

    def run(self):
        self.iter_curr += 1
        curr_pos = self.start_state
        currx, curry = curr_pos[0], curr_pos[1]
        while not(currx == -1 and curry == -1):
            currx, curry = self.expected_award(currx, curry)
        self.episode_reward.append(self.curr_total_reward/self.iter_curr)
    
    def print_policy(self):
        print("State Action Value are : ")
        print(self.action_value)
        print("Optimal Policy is : ")
        for i in range(self.maxx):
            for j in range(self.maxy):
                a = self.find_optimal_action(i, j)
                letter = 'L' if a == 0 else ('U' if a == 1 else ('R' if a == 2 else 'D'))
                print(letter, end = '')
            print()

def plot_graph(obj_list):
    for obj in obj_list:
        plt.plot(obj[0], '-', c = obj[1], label = obj[2])
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.legend(loc = 'best', prop = {'size':12})
    plt.show()

def main():
    obj = CliffWalking_sarsa(10, 100, 5, 1, 0.1, 0.1, 400)
    obj.sarsa()
    obj.print_policy()
    objql = CliffWalking_ql(10, 100, 5, 1, 0.1, 0.1, 400)
    objql.ql()
    objql.print_policy()
    plot_graph([[obj.episode_reward, 'red', 'sarsa'], [objql.episode_reward, 'blue', 'q-learning']])


if __name__ == '__main__':
    main()