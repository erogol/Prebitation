from __future__ import division

import random
import numpy as np
import pandas as pd 
from collections import namedtuple

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers import  Conv2D, Conv1D, Flatten, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD


def read_data(data_path, time_gran):
    df = pd.read_csv(data_path)
    # Set float64 to float32
    for c in df:
        if df[c].dtype == "float64":
            df[c] = df[c].astype('float32')
    # srot steps by time
    df.sort_values('Timestamp')
    # for TA-Lib
    df.rename(columns={'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close',
        'Volume_(BTC)': 'volume'}, inplace=True)
    # set time granularity
    num_steps = len(df[::time_gran])
    # set period data
    new_df = pd.DataFrame()
    new_df['timestamp'] = df['Timestamp'][::time_gran]
    new_df['high'] = df['high'].rolling(time_gran).max()
    new_df['low'] = df['low'].rolling(time_gran).min()
    new_df['close'] = df['close'][::time_gran]
    new_df['wprice'] = df['Weighted_Price'][::time_gran]
    new_df['open'] = df['open'].shift(time_gran-1)[::time_gran]
    assert new_df.shape[0] ==  num_steps
    # Logging
    print(" > There are {} rows".format(new_df.shape[0]))
    return new_df



def create_states(indata, n=50, test=False):
    close_prices = indata['close'].values
    high_prices = indata['high'].values 
    low_prices = indata['low'].values

    # generate time series per instance
    close_ts = generate_timeseries(close_prices, n)
    high_ts = generate_timeseries(high_prices, n)
    low_ts = generate_timeseries(low_prices, n)

    # normalize time series with the latest price value
    norm_val = close_ts[:, -1][:, None]
    high_ts /= norm_val
    low_ts /= norm_val
    close_ts /= norm_val

    # setup close prices
    close_prices = indata['close'].values[n:]
    assert close_ts.shape[0] == len(close_prices) 

    # structure data blobs
    feats = np.concatenate([close_ts[None, :],high_ts[None, :],low_ts[None, :]])
    feats = feats.transpose(1, 2, 0)

    # if test == False:
        # scaler = preprocessing.StandardScaler()
    #     feats = scaler.fit_transform(feats)
    #     joblib.dump(scaler, '../data/rl_scaler.pkl')
    # elif test == True:
    #     scaler = joblib.load('../data/rl_scaler.pkl')
    #     feats = scaler.transform(feats)
    return feats, close_prices


def generate_timeseries(prices, n):
    """Use the first time period to generate all possible time series of length n
       and their corresponding label.

    Args:
        prices: A numpy array of floats representing prices over the first time
            period.
        n: An integer representing the length of time series.

    Returns:
        A 2-dimensional numpy array of size (len(prices)-n) x n. Each row
        represents a time series of length n.
    """

    # ignore first n rows due to shortage of the history
    m = len(prices) - n
    ts = np.empty((m, n))
    for i in range(m):
        ts[i, :] = prices[i:i + n]
    return ts


def take_action(states, action, signal, time_step):
    terminal_state = 0

    #if it is the next state is the last state
    if time_step + 2 == states.shape[0]:
        terminal_state = 1
        signal.loc[time_step + 1] = action

    signal.loc[time_step] = action 
    time_step += 1
    #move the market data window one step forward
    next_state = states[time_step][None, :]
    
    return next_state, time_step, signal, terminal_state


def get_reward(time_step, prices, signals, eval=False):
    """ Reward considering only the positioning after each episode. It is analogous to predict price changes """
    reward = 0
    if eval == False:
        assert prices[time_step] >= 0 and prices[time_step-1] >= 0, "Price cannot be less then 0."
        # reward = np.log((prices[time_step] / prices[time_step-1]) * signals[time_step] + (1.-signals[time_step]))
        reward = 100 * (prices[time_step] - prices[time_step-1])/prices[time_step - 1] * signals[time_step]
    else:
        assert len(signals) == len(prices)
        # for t in range(len(signals)-1):  # ignore the last signal
        #     signal = signals[t]
        #     reward += np.log((float(prices[t]) / prices[t-1]) * signal + (1-signal))
        # reward /= float(len(signals))
    return reward


def create_model(num_actions, n=50):
    model = Sequential()
    model.add(Conv1D(2, 5, strides=1,
                     padding='valid', dilation_rate=1,
                     activation=None, use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros',
                     kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=None, kernel_constraint=None,
                     bias_constraint=None, input_shape=(n, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv1D(20, 46))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # model.add(Conv1D(3, 1, init='lecun_uniform'))
    model.add(Flatten())
    model.add(Dense(1, kernel_initializer='uniform'))
    model.add(Activation('sigmoid'))  # out in [0, 1]

    # rms = RMSprop()
    # sgd = SGD(0.001, momentum=0.85)
    adam = Adam(3e-5)
    model.compile(loss='mse', optimizer=adam)
    return model


def evaluate_Q(eval_states, eval_prices, eval_model, reward_func):
    #This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
    signals = pd.Series(index=np.arange(len(eval_prices)))
    time_step = 0
    state = eval_states[time_step][None, :]
    terminal_state = 0
    avg_reward = AverageMeter()
    while(terminal_state == 0):
        #We start in state S
        #Run the Q function on S to get predicted reward values on all the possible actions
        qval = eval_model.predict(state, batch_size=1)
        # print(qval)
        action = qval[0][0]
        #Take action, observe new state S'
        new_state, new_time_step, signals, terminal_state = take_action(eval_states, action, signals, time_step)
        #Observe reward
        eval_reward = reward_func(time_step, eval_prices, signals, eval=False)
        avg_reward.update(eval_reward)
        # update time window
        state = new_state
        time_step = new_time_step
    # counts of action_values
    # unique, counts = np.unique(filter(lambda v: v==v, signals.values), return_counts=True)
    return avg_reward.avg


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
    # max_newQ = np.max(newQ, 1)
    bellman = gamma * newQ + rewards[:, None]
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
    y_train[:] += bellman
    # print(y_train)
    # print("--")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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


class ActionMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None]*capacity

    def insert(self, idx, action):
        """Saves a transition."""
        self.memory.insert(idx, action)

    def get(self, idx):
        return self.memory[idx]

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