from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


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

def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))

def compute_features(df):
    df['CN'] = df.close.shift(-1) - df.close
    df['WC'] = df.wprice - df.close

    df['HO'] = df.high - df.open
    df['LO'] = df.low - df.open
    df['CO'] = df.close - df.open
    df['WO'] = df.wprice - df.open
    return df

def compute_labels(df):
    UP = (df['CN'].round(decimals=2)>0).astype(int)
    DN = (df['CN'].round(decimals=2)<0).astype(int)
    FLAT = np.logical_and(UP==0, DN==0).astype(int)
    df_Yt = pd.concat([UP, DN, FLAT], join = 'outer', axis =1)
    return df_Yt.values, ['UP', 'DN', 'FLAT']

def check_labels(labels, check_values):
  for idx, pred in enumerate(labels):
    if idx+1 == len(labels):
      break
    val = check_values[idx]
    valn = check_values[idx+1]
    if pred == 0:
        assert valn-val > 0, "idx: {}, pred: {}, val: {}, valn: {}".format(idx, pred, val, valn)
    elif pred == 1:
        assert valn-val < 0
    elif pred == 2:
        assert val-valn == 0

def model_performance(model, label_names, data, labels):
  Series_pred = np.argmax(model.predict(data, 
                                      batch_size=32, 
                                      verbose = 0),axis = 1)
  Series_actual = np.argmax(labels, axis = 1)
  classreport= classification_report(Series_actual, Series_pred, 
                                     target_names = label_names,
                                     digits = 4)
  print(classreport)

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap != None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100.0
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.show()