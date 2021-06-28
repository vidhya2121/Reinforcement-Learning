from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, ACTION_SPACE
from grid_world_print import print_values, print_policy
from td0 import rand_action, play_game


class Model:
    def __init__(self):
        self.theta = np.random.randn(4) / 2
    
    def s2x(self, s):
        return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3, 1])
    
    def predict(self, s):
        x = self.s2x(s)
        return self.theta.dot(x)
    
    def grad(self, s):
        return self.s2x(s)
    
    
    
if __name__ == '__main__':
    grid = standard_grid()
    print_values(grid.rewards, grid)
    print('\n\n')
    
    V = {}
    
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',
    }
    
    model = Model()
    k = 0.1
    
    for i in range(20000):
        if i % 10 == 0:
            k = k + 0.01
        alpha = 0.1 / k
        state_reward = play_game(grid, policy)
        for t in range(len(state_reward)-1):
            s, _ = state_reward[t]
            s_next, r = state_reward[t+1]
            old_theta = model.theta.copy()
            if grid.is_terminal(s_next):
                target = r
            else:
                target = r + 0.9 * model.predict(s_next)
            model.theta = model.theta + 0.1 * (target - model.predict(s)) * model.grad(s)
            
    for s in grid.all_states():
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            V[s] = 0
            
            
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)