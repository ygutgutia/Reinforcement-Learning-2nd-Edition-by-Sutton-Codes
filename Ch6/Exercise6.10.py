import numpy as np
import random
from matplotlib import pyplot as plt


class WindyGridworld:
    def __init__(self, rows, cols, wind_speed, action_map, disc_factor, expl_cons, step_size, iters, start_state, end_state):
        self.rows = rows
        self.cols = cols
        self.iters = iters
        self.action_map = action_map
        self.possible_actions = len(self.action_map)
        self.step_size = step_size
        self.expl_cons = expl_cons
        self.disc_factor = disc_factor
        self.start_state = start_state
        self.end_state = end_state
        self.episode_count = []
        self.wind_speed = wind_speed # cols size array
        self.action_value = np.zeros(shape=(rows, cols, self.possible_actions))

    def sarsa(self):
        self.currepisode = 0
        for i in range(self.iters):
            self.run(i+1)
            print("Iteration Done : " + str(i) + " Episodes Taken = "+ str(self.currepisode))
    
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

    def get_reward(self, currx, curry):
        if([currx, curry] == self.end_state):
            return 0
        return -1

    def get_new_state(self, currx, curry, action):
        wind_schotastic = np.random.uniform()
        curr_wind_speed = self.wind_speed[curry]
        if wind_schotastic < 0.33:
            curr_wind_speed += 1
        elif wind_schotastic >= 0.67:
            curr_wind_speed -= 1
        newx = currx + self.action_map[action][0] - curr_wind_speed # Shifts upwards
        newy = curry + self.action_map[action][1]
        newx = min(self.rows-1, max(0, newx))
        newy = min(self.cols-1, max(0, newy))
        return newx, newy

    def expected_award(self, currx, curry, action):
        self.currepisode += 1
        if(currx == self.end_state[0] and curry == self.end_state[1]):
            return -1, -1, -1
        if(currx < 0 or curry < 0 or currx >= self.rows or curry >= self.cols):
            return -1, -1, -1

        newx, newy = self.get_new_state(currx, curry, action)
        reward = self.get_reward(newx, newy)

        new_action = self.find_action(newx, newy)
        next_qsa = self.action_value[newx][newy][new_action]
        self.action_value[currx][curry][action] += self.step_size * (reward + \
                            next_qsa * self.disc_factor - self.action_value[currx][curry][action])
    
        return newx, newy, new_action

    def run(self, iter_curr):
        curr_pos = self.start_state
        currx, curry, action = curr_pos[0], curr_pos[1], self.find_action(curr_pos[0], curr_pos[1])
        while not(currx == -1 and curry == -1 and action == -1):
            currx, curry, action = self.expected_award(currx, curry, action)
        self.episode_count.append(self.currepisode/iter_curr)
        

def plot_graph(obj_list):
    for obj in obj_list:
        plt.plot(obj[0], '-', c = obj[1], label = obj[2])
    plt.xlabel("Iteration")
    plt.ylabel("Episodes")
    plt.legend(loc = 'best', prop = {'size':12})
    plt.show()

def main():
    # 0 - L, 1 - U, 2 - R, 3 - D
    action_map = [[-1, 0], [0, -1], [0, 1], [1, 0]]
    obj = WindyGridworld(7, 10, [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], action_map,
                            1, 0.1, 0.5, 1000, [3, 0], [3, 7])
    obj.sarsa()
    plot_graph([[obj.episode_count, 'red', '4-actions']])


if __name__ == '__main__':
    main()