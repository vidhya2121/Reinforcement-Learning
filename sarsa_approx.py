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


class Model:
    def __init__(self):
        self.theta = np.random.randn(25) / np.sqrt(25)
    def sa2x(self, s, a):
        return np.array([
          s[0] - 1              if a == 'U' else 0,
          s[1] - 1.5            if a == 'U' else 0,
          (s[0]*s[1] - 3)/3     if a == 'U' else 0,
          (s[0]*s[0] - 2)/2     if a == 'U' else 0,
          (s[1]*s[1] - 4.5)/4.5 if a == 'U' else 0,
          1                     if a == 'U' else 0,
          s[0] - 1              if a == 'D' else 0,
          s[1] - 1.5            if a == 'D' else 0,
          (s[0]*s[1] - 3)/3     if a == 'D' else 0,
          (s[0]*s[0] - 2)/2     if a == 'D' else 0,
          (s[1]*s[1] - 4.5)/4.5 if a == 'D' else 0,
          1                     if a == 'D' else 0,
          s[0] - 1              if a == 'L' else 0,
          s[1] - 1.5            if a == 'L' else 0,
          (s[0]*s[1] - 3)/3     if a == 'L' else 0,
          (s[0]*s[0] - 2)/2     if a == 'L' else 0,
          (s[1]*s[1] - 4.5)/4.5 if a == 'L' else 0,
          1                     if a == 'L' else 0,
          s[0] - 1              if a == 'R' else 0,
          s[1] - 1.5            if a == 'R' else 0,
          (s[0]*s[1] - 3)/3     if a == 'R' else 0,
          (s[0]*s[0] - 2)/2     if a == 'R' else 0,
          (s[1]*s[1] - 4.5)/4.5 if a == 'R' else 0,
          1                     if a == 'R' else 0,
          1
        ])
    
    def predict(self, s, a):
        x = self.sa2x(s, a)
        return self.theta.dot(x)

    def grad(self, s, a):
        return self.sa2x(s, a)
    
    
def getQs(model, s):
    Qs = {}
    for a in ACTION_SPACE:
        q = model.predict(s,a)
        Qs[a] = q
    return Qs

if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.1)
    idx = 0
    print_values(grid.rewards, grid)
    print('\n\n')
    
    SA2IDX = {}
    for s in grid.all_states():
        SA2IDX[s] = {}
        for a in ACTION_SPACE:
            SA2IDX[s][a] = idx
            idx = idx + 1
            
    model = Model()
    t = 1.0
    tt = 1.0
    deltas = []
    
    for it in range(20000):
        if it%100 == 0:
            t = t + 0.01
            tt = tt + 0.01
        if it%2000 == 0:
            print(it)
            
        alpha = 0.1/tt
            
        s = (2,0)
        grid.set_state(s)
        
        Qs = getQs(model, s)
        
        a = max_of(Qs)[0]
        a = rand_action(a, eps=0.5/t)
        biggest_change = 0
        while not grid.game_over():
            r = grid.move(a)
            s_next = grid.current_state()
            
            old_theta = model.theta.copy()
            if grid.is_terminal(s_next):
                model.theta += alpha * (r - model.predict(s,a)) * model.grad(s,a)
            else:
                Qs_next = getQs(model, s_next)               
                a_next = max_of(Qs_next)[0]
                a_next = rand_action(a_next, eps=0.5/t)     
                model.theta += alpha * (r + 0.9*model.predict(s_next, a_next )- model.predict(s,a)) * model.grad(s,a)
        
                s = s_next
                a = a_next
            biggest_change = max(biggest_change, np.abs(model.theta - old_theta).sum())
            
          
            
            
            
        deltas.append(biggest_change)
        
    plt.plot(deltas)
    plt.show()
        
    V = {}
    policy = {}
    for s in grid.actions.keys():
        Qs = getQs(model, s)
        a, q = max_of(Qs)
        policy[s] = a
        V[s] = q

    
            
            
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)