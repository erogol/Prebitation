import random
import joblib
import numpy as np
import bigfloat as bg
import pandas as pd
import scipy as sc

from numpy.linalg import norm
from talib.abstract import *
from matplotlib import pylab as plt
from collections import namedtuple

from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn import preprocessing

# import backtest as twp


def read_dummy_data1(data_path, time_gran):
    df = pd.read_csv(data_path)
    # Set float64 to float32
    for c in df:
        if df[c].dtype == "float64":
            df[c] = df[c].astype('float32')
    df.sort_values('Timestamp')
    # set time granularity
    df = df[::time_gran]
    # for TA-Lib
    df.rename(columns={'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close',
        'Volume_(BTC)': 'volume'}, inplace=True)
    # Logging
    print(" > There are {} rows".format(df.shape[0]))
    # price = np.arange(df.shape[0])[::-1]
    price = np.random.rand(df.shape[0])
    df['open'] = price
    df['high'] = price
    df['low'] = price
    df['close'] = price
    df['volume'] = [1] * df.shape[0]
    return df


def read_data(data_path, time_gran):
    df = pd.read_csv(data_path)
    # Set float64 to float32
    for c in df:
        if df[c].dtype == "float64":
            df[c] = df[c].astype('float32')
    df.sort_values('Timestamp')
    # set time granularity
    df = df[::time_gran]
    # for TA-Lib
    df.rename(columns={'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close',
        'Volume_(BTC)': 'volume'}, inplace=True)
    # Logging
    print(" > There are {} rows".format(df.shape[0]))
    return df


#Initialize first state, all items are placed deterministically
def create_states(indata, test=False):
    close = indata['close'].values
    diff = np.diff(close)
    diff = np.insert(diff, 0, 0)
    sma15 = SMA(indata.astype('f8'), timeperiod=15)  # simple moving average
    sma60 = SMA(indata.astype('f8'), timeperiod=60)
    rsi = RSI(indata.astype('f8'), timeperiod=14)
    atr = ATR(indata.astype('f8'), timeperiod=14)
    # ub, mb, lb = BBANDS(indata.astype('f8'), timeperiod=20, nbdevup=2, nbdevdn=2)
    # print(ub)
    # bbdiff = ub - lb

    feats = np.column_stack((close, diff, sma15, close-sma15, sma15-sma60, rsi, atr))
    feats = np.nan_to_num(feats)
    if test == False:
        scaler = preprocessing.StandardScaler()
        feats = scaler.fit_transform(feats)
        joblib.dump(scaler, '../data/rl_scaler.pkl')
    elif test == True:
        scaler = joblib.load('../data/rl_scaler.pkl')
        feats = scaler.transform(feats)
    return feats, close


def take_action(states, action, signal, time_step):
    #this should generate a list of trade signals that at evaluation time are fed to the backtester
    #the backtester should get a list of trade signals and a list of price data for the assett
    
    terminal_state = 0

    #if it is the next state is the last state
    if time_step + 2 == states.shape[0]:
        terminal_state = 1
        signal.loc[time_step + 1] = 0

    if action == 0:
        signal.loc[time_step] = 0 
    elif action == 1: 
        signal.loc[time_step] = 1 
    elif action == 2:
        signal.loc[time_step] = -1  
    else:
        raise RuntimeError("Unkown Action Idx")

    time_step += 1

    #move the market data window one step forward
    next_state = states[time_step][None, :]
    
    return next_state, time_step, signal, terminal_state


#Get Reward, the reward is returned at the end of an episode
def get_pos_reward(new_state, new_time_step, action, prices, signals, eval=False):
    """ Reward considering only the positioning after each episode. It is analogous to predict price changes """
    reward = 0
    # signal.fillna(value=0, inplace=True)
    if eval == False:
        assert prices[new_time_step] >= 0 and prices[new_time_step-1] >= 0, "Price cannot be less then 0."
        reward = (((float(prices[new_time_step]) / prices[new_time_step-1]) - 1) * signals[new_time_step-1])
    else:
        # signals.fillna(0)  # fill outsight time steps with zero
        assert len(signals) == len(prices)
        for t in range(len(signals)-1):  # ignore the last signal
            signal = signals[t]
            reward += ((float(prices[t+1]) - prices[t])) * signal
    return reward


def evaluate_Q(eval_data, eval_model, reward_func):
    #This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
    signals = pd.Series(index=np.arange(len(eval_data)))
    states, price_data = create_states(eval_data, test=True)
    time_step = 0
    state = states[time_step][None, :]
    terminal_state = 0
    while(terminal_state == 0):
        #We start in state S
        #Run the Q function on S to get predicted reward values on all the possible actions
        qval = eval_model.predict(state, batch_size=1)
        action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, new_time_step, signals, terminal_state = take_action(states, action, signals, time_step)
        #Observe reward
        eval_reward = reward_func(new_state, new_time_step, action, price_data, signals, eval=True)
        # update time window
        state = new_state
        time_step = new_time_step
    # counts of action_values
    unique, counts = np.unique(filter(lambda v: v==v, signals.values), return_counts=True)
    return eval_reward, np.asarray((unique, counts)).T


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def optimize_DQN(model, replay, batch_size, gamma):
    if (len(replay) < batch_size):
        return

    # from IPython.core.debugger import Tracer; Tracer()()
    memory = replay.sample(batch_size)
    batch = Transition(*zip(*memory))
    X_train = []
    y_train = []

    old_states = np.concatenate(batch.state)
    new_states = np.concatenate(batch.next_state)
    actions = np.array(batch.action)
    rewards = np.array(batch.reward)

    oldQ = model.predict(old_states, batch_size=batch_size)
    # print(old_states)
    # action_values = oldQ[:, actions]

    newQ = model.predict(new_states, batch_size=batch_size)
    max_newQ = np.max(newQ, 1)
    bellman = gamma * max_newQ + rewards
    # print(actions)
    # print(rewards)
    # print("--")


    # for memory in minibatch:
    #     old_state, action, new_state, reward = memory
    #     oldQ = model.predict(old_state, batch_size=1)
    #     newQ = model.predict(new_state, batch_size=1)
    #     max_newQ = np.max(newQ)
    #     y = np.zeros(oldQ.shape[1])
    #     # y[:] = oldQ[:]
    #     update = reward + (gamma * max_newQ)  # Bellman equation
    #     y[action] += update
    #     X_train.append(old_state)
    #     y_train.append(y[:])

    # X_train = np.squeeze(np.array(X_train), axis=(1))
    X_train = old_states
    y_train = np.zeros(oldQ.shape)
    # y_train = oldQ
    y_train[:, actions] += bellman
    # print(y_train)
    # print("--")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def is_full(self):
        return len(self.memory) == self.capacity

    def __len__(self):
        return len(self.memory)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import keras.backend as K

def huber_loss(y_true, y_pred, clip_value=1):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))