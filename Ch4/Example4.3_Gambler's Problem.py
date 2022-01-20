import numpy as np
from matplotlib import pyplot as plt

class GamblerProblem:
    def __init__(self, ph, max_capital, threshold):
        self.ph = ph
        self.threshold = threshold
        self.max_capital = max_capital
        self.state_value = np.zeros(max_capital + 1)
        self.optimal_policy = np.zeros(max_capital + 1)
        self.state_value[100] = 1

    def expected_reward(self, curr_state):
        action_state = [0 , min(curr_state, self.max_capital - curr_state)]
        new_state_value = 0
        optimal_next_state = 0
        for i in range(action_state[0], action_state[1]+1):
            dp = self.ph * self.state_value[curr_state + i] + (1 - self.ph) * self.state_value[curr_state - i]
            if(new_state_value < dp):
                new_state_value = dp
                optimal_next_state = i
        return new_state_value, optimal_next_state

    def value_iteration(self):
        iteration_count = 1
        while True:
            delta = 0
            for curr_state in range(1, self.max_capital):
                new_state_value, _ = self.expected_reward(curr_state)
                delta = max(delta, abs(self.state_value[curr_state] - new_state_value))
                self.state_value[curr_state] = new_state_value
            if(iteration_count in [1, 2, 3, 5, 10, 50, 100, 500, 1000]):
                plt.plot(self.state_value, label = 'Sweep '+str(iteration_count))
            print("Value Iteration Count = " + str(iteration_count) + " with delta = " + str(delta))
            iteration_count += 1
            if delta < self.threshold:
                break
    
    def policy_update(self):
        for curr_state in range(1, self.max_capital):
            _, optimal_next_state = self.expected_reward(curr_state)
            self.optimal_policy[curr_state] = optimal_next_state
    
    def display_state_value(self):
        plt.plot(self.state_value, label="Final Value Function")
        plt.xlabel('Capital')
        plt.ylabel('Value Estimate')
        plt.title('Value Function')
        plt.legend()
        plt.show()
    
    def display_optimal_policy(self):
        plt.bar(range(0, self.max_capital+1), self.optimal_policy)
        plt.xlabel('Capital')
        plt.ylabel('Final Policy(Stake')
        plt.title('Final Policy Function')
        plt.show()

# For ph = 0.4, 0.25 the graph isnt very unexpected but something intersting happens at ph=0.55
# Check out yourself
def main(ph = 0.55, max_capital = 100, threshold = 0.000001):
    obj = GamblerProblem(ph, max_capital, threshold)
    obj.value_iteration()
    obj.policy_update()
    obj.display_state_value()
    obj.display_optimal_policy()

if __name__ == '__main__':
    main()
