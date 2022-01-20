import numpy as np
from matplotlib import pyplot as plt

class BlackJack:
    def __init__(self, disc_factor, iters):
        # User sum(Max 21), Dealers showing card(Max 10), Usable Ace or not
        self.state_value = np.zeros(shape=(22, 11, 2))
        self.state_visit = np.zeros(shape=(22, 11, 2))
        self.iters = iters
        self.disc_factor = disc_factor

    def value_iteration(self):
        for i in range(self.iters):
            self.play_game()
            print("Iteration Done : " + str(i))
            # if(i in [10000, 50000, 100000]):
            #     self.plot_value_function()
    
    def expected_award(self, psum, dA, dsum, p_use_ace):
        if(psum >= 22 and dsum >= 22):
            return 0
        if(psum >= 22):
            return -1
        elif(dsum >= 22):
            return 1
        elif(psum >= 20 and dsum >= 17):
            reward = 1 if (psum > dsum) else (0 if (psum == dsum) else -1)
        elif(psum >= 20 and dsum < 17):
            d_hit = np.random.randint(1, 11)
            dsum_temp = (dsum + d_hit) if (d_hit != 1) else (dsum + d_hit + (10 if dsum <= 10 else 0))
            reward = self.expected_award(psum, dA, dsum_temp, p_use_ace) * self.disc_factor
        elif(psum < 20 and dsum >= 17):
            p_hit = np.random.randint(1, 11)
            p_use_ace_temp = (p_use_ace or (p_hit == 1)) if (psum <= 10) else p_use_ace
            psum_temp = psum + p_hit
            psum_temp += 0 if (psum <= 10 and p_hit != 1) else 10
            reward = self.expected_award(psum_temp, dA, dsum, p_use_ace_temp) * self.disc_factor
        else:
            d_hit = np.random.randint(1, 11)
            dsum_temp = (dsum + d_hit) if (d_hit != 1) else (dsum + d_hit + (10 if dsum <= 10 else 0))
            p_hit = np.random.randint(1, 11)
            p_use_ace_temp = (p_use_ace or (p_hit == 1)) if (psum <= 10) else p_use_ace
            psum_temp = psum + p_hit
            psum_temp += 0 if (psum <= 10 and p_hit != 1) else 10
            reward = self.expected_award(psum_temp, dA, dsum_temp, p_use_ace_temp) * self.disc_factor

        self.state_visit[psum][dA][p_use_ace] += 1
        self.state_value[psum][dA][p_use_ace] += (reward - self.state_value[psum][dA][p_use_ace]) \
                                                        / self.state_visit[psum][dA][p_use_ace]
        return self.state_value[psum][dA][p_use_ace]

    def play_game(self):
        pA = np.random.randint(1, 11)
        pB = np.random.randint(1, 11)
        dA = np.random.randint(1, 11) # showing
        dB = np.random.randint(1, 11)

        p_use_ace = (pA == 1 or pB == 1)
        d_use_ace = (dA == 1 or dB == 1)
        psum = 12 if (pA == 1 and pB == 1) else (pA + pB + (10 if p_use_ace else 0))
        dsum = 12 if (dA == 1 and dB == 1) else (dA + dB + (10 if d_use_ace else 0))
        self.expected_award(psum, dA, dsum, p_use_ace)

    def plot_value_function(self):
        X, Y, Z = [], [], []
        for i in range(12, self.state_value.shape[0]):
            for j in range(1, self.state_value.shape[1]):
                X.append(i)
                Y.append(j)
                Z.append(self.state_value[i][j][1])
        x = np.reshape(X, (10, 10))
        y = np.reshape(Y, (10, 10))
        z = np.reshape(Z, (10, 10))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('State Value for Usable Ace')
        plt.show()
        
        X, Y, Z = [], [], []
        for i in range(12, self.state_value.shape[0]):
            for j in range(1, self.state_value.shape[1]):
                X.append(i)
                Y.append(j)
                Z.append(self.state_value[i][j][0])
        x = np.reshape(X, (10, 10))
        y = np.reshape(Y, (10, 10))
        z = np.reshape(Z, (10, 10))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('State Value for Non Usable Ace')
        plt.show()
        

def main():
    obj = BlackJack(1, 50000)
    obj.value_iteration()
    obj.plot_value_function()

if __name__ == '__main__':
    main()