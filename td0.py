from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_world import standard_grid, ACTION_SPACE
from grid_world_print import print_values, print_policy

def rand_action(a, eps=0.2):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ACTION_SPACE)
    
    
def play_game(grid, policy):
    s = (2,0)
    grid.set_state(s)
    state_reward = [(s,0)]
    while not grid.game_over():
        action = rand_action(policy[s])
        reward = grid.move(action)
        s = grid.current_state()
        state_reward.append((s,reward))
    return state_reward

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
    
    for s in grid.all_states():
        V[s] = 0
        
    for i in range(1000):
        state_reward = play_game(grid, policy)
        for t in range(len(state_reward)-1):
            s, _ = state_reward[t]
            s_next, r = state_reward[t+1]
            V[s] = V[s] + 0.1 * (r + 0.9 * V[s_next] - V[s])
            
            
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
        
        
    
    