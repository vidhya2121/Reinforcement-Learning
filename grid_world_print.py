from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_world import windy_grid, ACTION_SPACE

SMALL_ENOUGH = 1e-3

def print_values(V, g):
    print(V)
    print('\n')
    for i in range(g.rows):
        print('-------------------------------------')
        for j in range(g.cols):
            v = V.get((i,j), 0)
            print("%.2f|" % v  + '\t', end="")
        print("")
            
def print_policy(P, g):
    print(P)
    print('\n')
    for i in range(g.rows):
        print('--------------------------------------')
        for j in range(g.cols):
            p = P.get((i,j), ' ')
            if p == ' ':
                print(' \t', end="")
            else:
                print(str(p) + '\t', end="")
        print("")
        
if __name__ == '__main__':
    rewards = {}
    grid = windy_grid()
    
    transition_probs = {}
    for (s,a), v in grid.probs.items():
        for s_next,p in v.items():
            transition_probs[(s,a,s_next)] = p
            rewards[(s,a,s_next)] = grid.rewards.get(s_next, 0)
            
    print(transition_probs)
    print('---------------')
    print(rewards)
    
    
    policy = {
    (2, 0): {'U': 0.5, 'R': 0.5},
    (1, 0): {'U': 1.0},
    (0, 0): {'R': 1.0},
    (0, 1): {'R': 1.0},
    (0, 2): {'R': 1.0},
    (1, 2): {'U': 1.0},
    (2, 1): {'R': 1.0},
    (2, 2): {'U': 1.0},
    (2, 3): {'L': 1.0},
    }
    print_policy(policy, grid)
    
    V = {}
    for s in grid.all_states():
        V[s] = 0
        gamma = 0.9
        
    it = 0
    while True:
        big_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_V = V[s]
                new_V = 0
                
                for a in ACTION_SPACE:
                    for s_next in grid.all_states():
                        action_prob = policy[s].get(a,0)
                        r = rewards.get((s, a, s_next),0)
                        new_V = new_V + action_prob * transition_probs.get((s,a,s_next), 0) * (r + gamma * V[s_next] )
                V[s] = new_V
                big_change = max(big_change, np.abs(old_V- V[s]))
                        
        print(it, big_change)
        print_values(V, grid)
                                
        it += 1
                        
        if big_change < SMALL_ENOUGH:
            break
