import numpy as np
from matplotlib import pyplot as plt

# TODO(@ygutgutia): TD Learning is giving different graph than MC(MC is correct) not sure why
class TD:
    def __init__(self, disc_factor, step_size, iters, max_state):
        self.disc_factor = disc_factor
        self.step_size = step_size
        self.iters = iters
        self.max_state = max_state
        self.state_value = np.full(shape=(max_state), fill_value=0.5)
        # Dynamics of the environment
        self.start_state = (int)(max_state/2)
        self.end_state = [0, max_state-1]

    def value_iteration(self):
        for i in range(self.iters):
            print("TD0 Step " + str(i))
            if(i in [0, 1, 10, 50, 100, self.iters-1]):
                plt.plot(range(1, self.max_state-1), self.state_value[1:self.max_state-1], label = 'Sweep '+str(i))
            self.expected_reward(self.start_state)
        plot_graph("TD Learning", self.max_state)

    def expected_reward(self, curr_state):
        if curr_state in self.end_state:
            return
        prob = np.random.uniform()
        next_state = curr_state+1 if prob < 0.5 else curr_state-1
        reward = 1 if next_state == self.end_state[1] else 0
        new_state_value = self.state_value[next_state]
        self.state_value[curr_state] += self.step_size * (reward + self.disc_factor * new_state_value - self.state_value[curr_state])
        self.expected_reward(next_state)
        return


class MC:
    def __init__(self, disc_factor, step_size, iters, max_state):
        self.disc_factor = disc_factor
        self.step_size = step_size
        self.iters = iters
        self.max_state = max_state
        self.state_value = np.full(shape=(max_state), fill_value=0.5)
        # Dynamics of the environment
        self.start_state = (int)(max_state/2)
        self.end_state = [0, max_state-1]

    def value_iteration(self):
        for i in range(self.iters):
            print("MC Step " + str(i))
            if(i in [0, 1, 10, 50, 100, self.iters-1]):
                plt.plot(range(1, self.max_state-1), self.state_value[1:self.max_state-1], label = 'Sweep '+str(i))
            self.expected_reward(self.start_state)
        plot_graph("Monte Carlo Learning", self.max_state)

    def expected_reward(self, curr_state):
        if curr_state == self.end_state[0]:
            return 0, 0
        if curr_state == self.end_state[1]:
            return 1, 0
        prob = np.random.uniform()
        reward, new_state_value = self.expected_reward(curr_state+1) if prob < 0.5 else self.expected_reward(curr_state-1)
        self.state_value[curr_state] += self.step_size * (reward + self.disc_factor * new_state_value - self.state_value[curr_state])
        return 0, self.state_value[curr_state]

def plot_graph(title, max_state):
    plt.plot(range(1, max_state-1), np.linspace(1/(max_state-1), 1, max_state-2), label = 'True Value')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    tdobj = TD(1, 0.1, 200, 7)
    tdobj.value_iteration()
    mcobj = MC(1, 0.1, 200, 7)
    mcobj.value_iteration()

if __name__ == '__main__':
    main()
