from __future__ import print_function, division
from builtins import range

import matplotlib.pyplot as plt
import numpy as np
from grid_world import standard_grid, ACTION_SPACE
from grid_world_print import print_values, print_policy
from monte_carlo_random import random_action, play_game


SMALL_ENOUGH = 1e-2
GAMMA = 0.9

LEARNING_RATE = 0.001

if __name__ == '__main__':
    grid = standard_grid()
    
    print_values(grid.rewards, grid)
    print('\n\n')
    
    V = {}
    returns = {}
    
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
    
    thetha = np.random.randn(4) / 2
    print(thetha)

    def s2x(s):
        return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] -3, 1])
    
    deltas = []
    t = 1.0
    for it in range(40000):
        if it % 100 == 0:
            t += 0.01
        alpha = LEARNING_RATE/t
        
        big_change = 0
        state_return = play_game(grid, policy)
        seen = set()
        for s, ret in state_return:
            if s not in seen:
                old_thetha = thetha.copy()
                x = s2x(s)
                
                v_hat = thetha.dot(x)
                thetha = thetha + alpha * (ret - v_hat) *x
                big_change = max(big_change, np.abs(old_thetha - thetha).sum())
                seen.add(s)
                
        deltas.append(big_change)
        
    plt.plot(deltas)
    plt.show()
        
    V = {}
    for s in grid.all_states():
        if s in grid.actions:
            V[s] = thetha.dot(s2x(s))
        else:
            V[s] = 0
            
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
    