import numpy as np
import random
from matplotlib import pyplot as plt

class BlackJack:
    def __init__(self, disc_factor, iters, expl_cons):
        # User sum(Max 21), Dealers showing card(Max 10), Usable Ace or not
        self.policy = np.ones(shape=(22, 11, 2), dtype=int)
        self.action_value = np.zeros(shape=(22, 11, 2, 2))
        self.state_visit = np.zeros(shape=(22, 11, 2, 2))
        self.iters = iters
        self.expl_cons = expl_cons
        self.disc_factor = disc_factor

    def value_iteration(self):
        for i in range(self.iters):
            self.play_game()
            print("Iteration Done : " + str(i))
    
    def expected_award(self, psum, dA, dsum, p_use_ace):
        if(psum >= 22 and dsum >= 22):
            return 0, 0
        if(psum >= 22):
            return -1, 0
        elif(dsum >= 22):
            return 1, 0
        
        dsum_temp = dsum
        if(dsum < 17):
            d_hit = np.random.randint(1, 11)
            dsum_temp = (dsum + d_hit) if (d_hit != 1) else (dsum + d_hit + (10 if dsum <= 10 else 0))

        player_prob = random.uniform(0, 1)
        optimal_action = self.policy[psum][dA][p_use_ace]
        if(player_prob < self.expl_cons):
            optimal_action = 0 if optimal_action == 1 else 1
        
        p_use_ace_temp = p_use_ace
        psum_temp = psum
        if(optimal_action == 1):
            p_hit = np.random.randint(1, 11)
            p_use_ace_temp = int((p_use_ace == 1) or (p_hit == 1)) if (psum <= 10) else p_use_ace
            psum_temp = psum + p_hit
            psum_temp += 0 if (psum <= 10 and p_hit != 1) else 10
        
        reward = 1 if (psum > dsum) else (0 if (psum == dsum) else -1)
        if(not(psum_temp == psum and dsum_temp == dsum)):
            reward, Gnext = self.expected_award(psum_temp, dA, dsum_temp, p_use_ace_temp)
            reward += Gnext * self.disc_factor

        self.state_visit[psum][dA][p_use_ace][optimal_action] += 1
        self.action_value[psum][dA][p_use_ace][optimal_action] += (reward - self.action_value[psum][dA][p_use_ace][optimal_action]) \
                                                        / self.state_visit[psum][dA][p_use_ace][optimal_action]
        self.policy[psum][dA][p_use_ace] = self.action_value[psum][dA][p_use_ace][1] > self.action_value[psum][dA][p_use_ace][0]
        return 0, self.action_value[psum][dA][p_use_ace][optimal_action]

    def play_game(self):
        pA = np.random.randint(1, 11)
        pB = np.random.randint(1, 11)
        dA = np.random.randint(1, 11) # showing
        dB = np.random.randint(1, 11)

        p_use_ace = (pA == 1 or pB == 1)
        d_use_ace = (dA == 1 or dB == 1)
        psum = 12 if (pA == 1 and pB == 1) else (pA + pB + (10 if p_use_ace else 0))
        dsum = 12 if (dA == 1 and dB == 1) else (dA + dB + (10 if d_use_ace else 0))
        self.expected_award(psum, dA, dsum, int(p_use_ace))

    def plot_optimal_policy(self):
        
        for i in range(12, self.policy.shape[0]):
            for j in range(1, self.policy.shape[1]):
                if(self.policy[i][j][1] == 1):
                    plt.scatter(j, i, c='g', s=5)
                else:
                    plt.scatter(j, i, c='r', s=5)
        plt.xlabel('Dealer Showing')
        plt.ylabel('Player Sum')
        plt.title('Usable Ace')
        plt.show()
        
        for i in range(12, self.policy.shape[0]):
            for j in range(1, self.policy.shape[1]):
                if(self.policy[i][j][0] == 1):
                    plt.scatter(j, i, c='g', s=5)
                else:
                    plt.scatter(j, i, c='r', s=5)
        plt.xlabel('Dealer Showing')
        plt.ylabel('Player Sum')
        plt.title('Unusable Ace')
        plt.show()
        

def main():
    obj = BlackJack(1, 500000, 1)
    obj.value_iteration()
    obj.plot_optimal_policy()

if __name__ == '__main__':
    main()