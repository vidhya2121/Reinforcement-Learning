from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_world import standard_grid, ACTION_SPACE
from grid_world_print import print_values, print_policy

def random_action(a):
    p = np.random.random()
    if p < 0.5:
      return a
    else:
      tmp = list(ACTION_SPACE)
      tmp.remove(a)
      return np.random.choice(tmp)


def play_game(grid, policy):
    
    start_state = list(grid.actions.keys())
    start_i = np.random.choice(len(start_state))
    grid.set_state(start_state[start_i])    
    
    s = grid.current_state()
    state_rewards = [(s,0)]
    
    
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        state_rewards.append((s,r))
        
   
        
    state_returns = []
    ret = 0
    first = True
    for s,r in reversed(state_rewards):
        if first:
            first = False
        else:
            state_returns.append((s,ret))
        ret = r + 0.9 * ret
     
    state_returns.reverse()
    return state_returns

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

    
    
    for s in grid.all_states():
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0
            
    for n in range(100):
        states_returns = play_game(grid, policy)
        seen = set()
        for s, ret in states_returns:
            if s not in seen:
                returns[s].append(ret)
                V[s] = np.mean(returns[s])
                seen.add(s)
                
    print("values:")
    print_values(V, grid)
    
    