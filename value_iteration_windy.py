from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_world import windy_grid, windy_grid_penalized, ACTION_SPACE
from grid_world_print import print_values, print_policy

GAMMA = 0.9 
SMALL_ENOUGH = 1e-3

def get_tp_rewards(grid):
    rewards = {}    
    transition_probs = {}
    for (s,a), v in grid.probs.items():
        for s_next,p in v.items():
            transition_probs[(s,a,s_next)] = p
            rewards[(s,a,s_next)] = grid.rewards.get(s_next, 0)
            
    
    return transition_probs, rewards

if __name__ == '__main__':
    
    grid = windy_grid()
    
    transition_probs, rewards = get_tp_rewards(grid)
    
    print('---------------------------------------------------')
    print(transition_probs)
    print('----------------r----------------------------------')
    print_values(grid.rewards, grid)
    
    
    V = {}
    for s in grid.all_states():
        V[s] = 0
    it = 0    
    while True:
        big_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = float('-inf')
                for a in ACTION_SPACE:
                    v = 0
                    for s_next in grid.all_states():
                        r = rewards.get((s, a, s_next),0)
                        v += transition_probs.get((s,a,s_next),0) * (r + GAMMA * V[s_next])
                    if v > new_v:
                        new_v = v
                V[s] = new_v
                
                big_change = max(big_change, np.abs(old_v- V[s]))
                            
                print(it, big_change)
                print_values(V, grid)
                                
        it += 1
                        
        if big_change < SMALL_ENOUGH:
            break
                   
    
    print('\n\n\n')
    
    policy = {}
    for s in grid.actions.keys():
        best_a = None
        best_v = float('-inf')
        for a in ACTION_SPACE:
            v = 0
            for s_next in grid.all_states():
                r = rewards.get((s, a, s_next),0)
                v += transition_probs.get((s,a,s_next),0) * (r + GAMMA * V[s_next])
            if v > best_v:
                best_v = v
                best_a = a
        policy[s] = best_a
    
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)