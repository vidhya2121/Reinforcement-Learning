from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_world import standard_grid, ACTION_SPACE
from grid_world_print import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9           
        
def get_prob_rewa(grid):
    transition_probs = {}
    rewards = {}    
    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s_next = grid.get_next_state(s, a)
                    transition_probs[(s, a, s_next)] = 1
                    if s_next in grid.rewards:
                        rewards[(s, a, s_next)] = grid.rewards[s_next]
    return transition_probs, rewards

def evaluate(grid, policy):
    V = {}
    for s in grid.all_states():
        V[s] = 0
        
    it = 0
    while True:
        big_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_V = V[s]
                new_V = 0
                
                for a in ACTION_SPACE:
                    for s_next in grid.all_states():
                        action_prob = 1 if policy.get(s) == a else 0
                        r = rewards.get((s, a, s_next),0)
                        new_V = new_V + action_prob * transition_probs.get((s,a,s_next), 0) * (r + GAMMA * V[s_next] )
                V[s] = new_V
                big_change = max(big_change, np.abs(old_V- V[s]))
                        
        #print(it, big_change)
        #print_values(V, grid)
                                
        it += 1
                        
        if big_change < SMALL_ENOUGH:
            break
    return V

if __name__ == '__main__':
        
    grid = standard_grid()
    transition_probs, rewards = get_prob_rewa(grid)
   
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTION_SPACE)
    
    print('---------------------------------------------------')
    print(transition_probs)
    print('----------------r----------------------------------')
    print_values(grid.rewards, grid)
    
                   
     # initial policy
    print("initial policy:")
    print_policy(policy, grid)
    
    print('\n\n\n')
    
    while True:
        V = evaluate(grid, policy)
        print(V)
        is_policy_optimized = True
        
        for s in grid.actions.keys():
            old_a = policy[s]
            print(old_a)
            new_a = None
            best_v = float('-inf')
            for a in ACTION_SPACE:
                v = 0
                for s_next in grid.all_states():
                    r = rewards.get((s, a, s_next),0)
                    v = v + transition_probs.get((s,a,s_next), 0) * (r + GAMMA * V[s_next] )
                
                if v > best_v:
                    best_v = v
                    new_a = a
            policy[s] = new_a
            if(old_a != new_a):
                is_policy_optimized = False
        if is_policy_optimized:
            break
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
    
    
    