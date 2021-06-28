from __future__ import print_function, division
from builtins import range

import numpy as np
import pandas as pd
import argparse
import itertools

from datetime import datetime
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# To run the program
# training - python stock_prices.py -m train && python plots.py -m train
# test - python stock_prices.py -m test && python plots.py -m test


# To read the stock prices
def get_data():
    df = pd.read_csv('aapl_msi_sbux.csv')
    return df.values

# To normalize the states
# Fill up the states by selecting a random action and executing
def get_scaler(env):
    print(env)
    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, action, info, done = env.step(action)
        states.append(state)
        if done:
            break
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

# To create a directory
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
# Linear Regression with Gradient Descent and Momentum
class LinearModel:
    # random initialization of W and b acc to shape
    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)
        
        self.mW = 0
        self.mb = 0
        
        self.losses = []
        
    # y = Wx + b
    def predict(self, X):
        return X.dot(self.W) + self.b
    
    # gradient descent with momentum and compute loss    
    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        num_values = np.prod(Y.shape)
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values
        
        self.mW = momentum * self.mW - learning_rate * gW
        self.mb = momentum * self.mb - learning_rate * gb
        
        self.W += self.mW
        self.b += self.mb
        
        mse = np.mean((Yhat - Y) ** 2)
        self.losses.append(mse)
        
    # load W and b from a file
    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']
        
    # save W and b to a file
    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)
        
        
class MultiStockEnv:
    # 3 stock trading
    # state : [shares1,shares2,shares3,price1,price2,price3,cash]
    # action : [sell,hold,buy]
        # [0,0,0] [0,0,1] [0,1,0] ... [2,0,0] ... (27 combo)
    
    def __init__(self, data, initial_investment=20000):
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape
        self.initial_investment = initial_investment
        self.curr_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        
        self.action_space = np.arange(3**self.n_stock)
        self.action_list = list(map(list, itertools.product([0,1,2], repeat=self.n_stock)))
        
        self.state_dim = self.n_stock * 2 + 1
        
        self.reset()
        
    def reset(self):
        self.curr_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.curr_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()
    
    def step(self, action):
        prev_val = self._get_val()
        self.curr_step += 1
        self.stock_price = self.stock_price_history[self.curr_step]
        self._trade(action)
        
        curr_val = self._get_val()
        reward = curr_val - prev_val
        
        done = self.curr_step == self.n_step - 1
        info = { 'cur_val': curr_val }
        
        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price 
        obs[-1] = self.cash_in_hand 
        return obs
    
    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand 
    
    def _trade(self, action):
        # 0, 1, 2 -> sell, hold, buy
        action_vec = self.action_list[action]
        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)
                
            if sell_index:
                for i in sell_index:
                    self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                    self.stock_owned[i] = 0
            if buy_index:
                can_buy = True
                while can_buy:
                    for i in buy_index:
                        if self.stock_price[i] < self.cash_in_hand:
                            self.stock_owned[i] += 1
                            self.cash_in_hand -= self.stock_price[i]
                        else:
                            can_buy = False
    
class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)
            target_full = self.model.predict(state)
            target_full[0,action] = target
            
            self.model.sgd(state, target_full)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)
        
def play_one_episode(agent, env, is_train):
        state = env.reset()
        state = scaler.transform([state])
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = scaler.transform([next_state])
            if is_train == True:
                agent.train(state, action, reward, next_state, done)
            state = next_state
        return info['cur_val']
            
        
if __name__ == '__main__':
    models_folder = 'stock_model'
    rewards_folder = 'stock_rewards'
    
    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
    print(parser)
    args = parser.parse_args()
    make_dir(models_folder)
    make_dir(rewards_folder)
    
    data = get_data()
    n_timestamps, n_stocks = data.shape
    
    n_train = n_timestamps // 2
    
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    
    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)
    print(args)
    portfolio_val = []
    
    if args.mode == 'test':
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        env = MultiStockEnv(test_data, initial_investment)
        
        agent.epsilon = 0.01
        agent.load(f'{models_folder}/linear.npz')
        
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, args.mode)
        
        dt = datetime.now() - t0
        
        print(f"episode: {e+1}/{num_episodes}, episode end value : {val:.2f}, duration: {dt}")
    
    
        portfolio_val.append(val)
        
    if args.mode == 'train':
        agent.save(f'{models_folder}/linear.npz')
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    
    
        plt.plot(agent.model.losses)
        plt.show()
        
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_val)