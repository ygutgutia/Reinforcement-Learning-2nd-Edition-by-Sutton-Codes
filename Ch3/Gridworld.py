# Converges approximately to the book example
# Using one-step policy update or TD(0)

import numpy as np
import random

# 0 - L, 1 - U, 2 - R, 3 - D
class Agent:
    def __init__(self, maxx, maxy, iters, episodes, discount):
        self.maxx = maxx
        self.maxy = maxy
        self.currx = 0
        self.curry = 0
        self.discount = discount
        self.iters = iters
        self.episodes = episodes
        self.state_value = np.zeros(shape=(maxx, maxy))
        self.state_visit = np.zeros(shape=(maxx, maxy))

    def reset(self):
        self.currx = np.random.choice(self.maxx)
        self.curry = np.random.choice(self.maxy)

    def run(self):
        for i in range(self.iters):
            for j in range(self.episodes):
                self.take_step()
            print("State Value after " + str(i+1) + " iterations: ")
            print(self.state_value)
            self.reset()

    def take_step(self):
        # all random policy    
        action = np.random.choice(4)
        self.move(action)
        
    def move(self, action):
        next_x = self.currx
        next_y = self.curry
        if(action == 0):
            next_x += -1
        elif(action == 1):
            next_y += -1
        elif(action == 2):
            next_x += 1
        elif(action == 3):
            next_y += 1

        if(next_x < 0 or next_x >= self.maxx or next_y < 0 or next_y >= self.maxy):
            self.update_state(self.currx, self.curry, -1)
        else:
            self.update_state(next_x, next_y, 0)
        
        if(next_x==0 and next_y==1):
            self.update_state(4, 1, 10)        
        if(next_x==0 and next_y==3):
            self.update_state(2, 3, 5)
    
    def update_state(self, next_x, next_y, reward):
        self.state_visit[self.currx][self.curry] += 1
        self.state_value[self.currx][self.curry] += (self.state_value[next_x][next_y] * self.discount + reward - 
                        self.state_value[self.currx][self.curry]) / self.state_visit[self.currx][self.curry]
        self.currx = next_x
        self.curry = next_y

def main():
    agent_obj = Agent(5, 5, 100, 1000, 0.9)
    agent_obj.run()
    print(agent_obj.state_value)
        
if __name__ == '__main__':
    main()
