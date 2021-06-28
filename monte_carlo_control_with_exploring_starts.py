from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_world import standard_grid,negative_grid, ACTION_SPACE
from grid_world_print import print_values, print_policy
import matplotlib.pyplot as plt

def play_game(grid, policy):
    
    start_state = list(grid.actions.keys())
    start_i = np.random.choice(len(start_state))
    grid.set_state(start_state[start_i])    
    
    s = grid.current_state()
    a = np.random.choice(ACTION_SPACE)
    state_action_rewards = [(s,a,0)]
    
    print('rewards')
    print(state_action_rewards)
    print('\n\n')
    seen_set = set()
    seen_set.add(grid.current_state())
    steps = 0
    
    while True:
        r = grid.move(a)
        steps += 1
        s = grid.current_state()
        
        if s in seen_set:
            reward = -10. / steps
            state_action_rewards.append((s,None,reward))
            break
        elif grid.game_over():
            state_action_rewards.append((s,None,r))
            break
        else:
            a = policy[s]
            state_action_rewards.append((s,a,r))
        seen_set.add(s)
        
    
    
    print('rewards')
    print(state_action_rewards)
    print('\n\n')
        
    state_action_returns = []
    ret = 0
    first = True
    for s,a,r in reversed(state_action_rewards):
        if first:
            first = False
        else:
            state_action_returns.append((s,a,ret))
        ret = r + 0.9 * ret
     
    state_action_returns.reverse()
    return state_action_returns
    
def max_of(d):
    max_val = float('-inf')
    max_key = None
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.9)

    
    print_values(grid.rewards, grid)
    print('\n\n')
    
    policy = {}
    for s in grid.actions.keys():
      policy[s] = np.random.choice(ACTION_SPACE)
    
    print_policy(policy, grid)
    Q = {}
    returns = {}
    
    for s in grid.all_states():
        if s in grid.actions:
            Q[s] = {}
            for a in ACTION_SPACE:
                Q[s][a] = 0
                returns[(s,a)] = []
        else:
            pass
        
    print(Q)
    print(returns)
    
    deltas = []
    
    for t in range(2000):
        if t % 100 == 0:
            print(t)
        state_action_returns = play_game(grid, policy)
        big_change = 0
        seen_state = set()
        
        for s,a,ret in state_action_returns:
            if s not in seen_state:
                old_q = Q[s][a]
                returns[(s,a)].append(ret)
                Q[s][a] = np.mean(returns[(s,a)])
                big_change = max(big_change, np.abs(old_q - Q[s][a]))
                seen_state.add(s)
        deltas.append(big_change)
        
        for s in policy.keys():
            policy[s] = max_of(Q[s])[0]
    
    plt.plot(deltas)
    plt.show()
    print("final policy:")
    print_policy(policy, grid)
    
    
    V = {}
    for s, Qofs in Q.items():
        V[s] = max_of(Q[s])[1]
        
        
    print("final values:")
    print_values(V, grid)