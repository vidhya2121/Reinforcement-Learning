from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid, ACTION_SPACE
from grid_world_print import print_values, print_policy

def rand_action(a, eps=0.2):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ACTION_SPACE)
    
def max_of(d):
    max_val = float('-inf')
    max_key = None
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.1)

    print_values(grid.rewards, grid)
    print('\n\n')
    
    Q = {}
    for s in grid.all_states():
        Q[s] = {}
        for a in ACTION_SPACE:
            Q[s][a] = 0
    
    update_count = {}
    t = 1.0
    deltas = []
    
    for it in range(10000):
        if it%100 == 0:
            t = t + 1e-2
        if it%2000 == 0:
            print(it)
            
        s = (2,0)
        grid.set_state(s)
        
        a = max_of(Q[s])[0]
        
        biggest_change = 0
        while not grid.game_over():
            a = rand_action(a, eps=0.5/t)
            r = grid.move(a)
            s_next = grid.current_state()
            a_next, q_next = max_of(Q[s_next])
            old_qsa = Q[s][a]
            Q[s][a] = Q[s][a] + 0.1*(r + 0.9*Q[s_next][a_next] - Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))
            
            update_count[s] = update_count.get(s,0) + 1
            
            s = s_next
            a = a_next
            
        deltas.append(biggest_change)
        
    plt.plot(deltas)
    plt.show()
        
    V = {}
    policy = {}
    for s in grid.actions.keys():
        a, q = max_of(Q[s])
        policy[s] = a
        V[s] = q
        
    print("update counts:")
    total = np.sum(list(update_count.values()))
    for k, v in update_count.items():
        update_count[k] = float(v) / total
        print_values(update_count, grid)
    
            
            
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)