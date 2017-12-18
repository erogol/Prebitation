import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, scale


class DatasetV1(object):
    
    
    def __init__(self, data_path, train_start, test_start, train_end=None, test_end=None, time_gran=10):
        self.test_start = test_start
        self.test_end = test_end
        self.train_start = train_start
        self.train_end = train_end
        self.time_gran = time_gran  # in minutes
        self.data_path = data_path
        # Read the data
        self.df = pd.read_csv(data_path)
        # Set float64 to float32
        for c in self.df:
            if self.df[c].dtype == "float64":
                self.df[c] = self.df[c].astype('float32')
        self.preprocessor = None
        # Define readable time
        self.df['Date'] = pd.to_datetime(self.df['Timestamp'],unit='s').dt.date
        self.df['Time'] = pd.to_datetime(self.df['Timestamp'],unit='s').dt.time
        # Sort by time
        self.df.sort_values('Timestamp')
        # Set time granularity
        self.df = self.df[0::self.time_gran]
#         self.group = self.df.groupby('Date')
        # Logging
        print(" > There are {} rows".format(self.df.shape[0]))
        print(self.df.head())
        
        
    def create_train_data_v1(self):
        """ Predict the next price difference given the previous price """
        self.split_data(self.tstart)
        self.X_train, self.y_train = self.__preprocess_data_v1(self.df_train)
        self.X_test, self.y_test = self.__preprocess_data_v1(self.df_test)
    
    
    def create_train_data_v2(self, feature_cols, label_col):
        """ Predict the next price difference given the previous row,
        Each row is a collection of difference values comparing to previous
        time step."""
        self.split_data(self.tstart)
        self.X_train, self.y_train = self.__preprocess_data_v2(self.df_train,
                                                               feature_cols,
                                                               label_col)
        self.X_test, self.y_test = self.__preprocess_data_v2(self.df_test,
                                                             feature_cols,
                                                             label_col)
        
        
    def create_train_data_v3(self, feature_cols, label_col, time_steps, preprocess):
        """ Predict the next price difference given the previous row,
        Each row is a collection of difference values comparing to previous
        time step."""
        self.split_data(self.train_start, self.test_start)
        self.df_train = self.__preprocess_data_v3(self.df_train,
                                                   feature_cols,
                                                   label_col,
                                                   time_steps,
                                                   preprocess)
        self.df_test = self.__preprocess_data_v3(self.df_test,
                                                 feature_cols,
                                                 label_col,
                                                 time_steps,
                                                 preprocess)
        self.X_train = self.df_train.values[:, 1:-1]
        self.y_train = self.df_train.values[:, -1]
        self.X_test = self.df_test.values[:, 1:-1]
        self.y_test = self.df_test.values[:, -1]
        
    
    def create_train_data_v4(self, time_steps, preprocess):
        """ Predict the next price difference given the previous row,
        Each row is a collection of difference values comparing to previous
        time step."""
        self.split_data(self.train_start, self.test_start)
        prices_train, v_bid_train, v_ask_train = self.__preprocess_data_v4(self.df_train,
                                                   time_steps,
                                                   preprocess)

        self.prices_train1, self.prices_train2 = np.split(prices_train, 2)
        self.v_bid_train1, self.v_bid_train2 = np.split(v_bid_train, 2)
        self.v_ask_train1, self.v_ask_train2 = np.split(v_ask_train, 2)

        self.prices_test, self.v_bid_test, self.v_ask_test = self.__preprocess_data_v4(self.df_test,
                                                 time_steps,
                                                 preprocess)


    def split_data(self, train_start, test_start, train_end=None, test_end=None):
        """ Split data into train and test with
        given start and end points of test split
        """
        if train_end is None and train_end is None:
            idx = self.df.shape[0]-test_start
            self.df_train= self.df[train_start:idx]
            self.df_test= self.df[idx:]
            print(" Train data size {}".format(self.df_train.size))
            print(" Test data size {}".format(self.df_test.size))
        else:
            raise NotImplemented()
        return self.df_train, self.df_test
    
    
    def __preprocess_data_v1(self, df):
        """
            1. Compute the price change values
            2. Apply MaxMinScaler 
            3. Split data and labels
            
            labels are the one step ahead values of the
            train values.
        """
        close_price = df['Close'].values.flatten()
        data = self.difference(close_price)  # trend removal
        data = data[:, None]
        assert data[0] == close_price[1] - close_price[0]
        assert len(data) ==  len(close_price) - 1

        # MaxMin scale input for NN
        if self.preprocessor:
            data = self.preprocessor.transform(data)
        else:
            self.preprocessor = MinMaxScaler() 
            data = self.preprocessor.fit_transform(data)

        # set train set and labels
        labels = data[1:len(data)]
        data = data[0:len(data)-1]
        data = np.reshape(data, (len(data), 1, 1))
        return data, labels
    
    
    def __preprocess_data_v2(self, df, feature_columns, label_column):
        """
            1. Compute difference btw successive rows for each feature column
            2. Apply MaxMinScaler 
            3. Split data and labels by the give feature column
            
            labels are the one step ahead values of the
            train values.
        """
        data = np.zeros([df.shape[0]-1, len(feature_columns)])
        labels = None
        label_col_idx = None
        for idx, col in enumerate(feature_columns):
            vals = df[col].values.flatten()
            diff_vals = self.difference(vals)
            assert diff_vals[0] == vals[1] - vals[0]
            assert len(diff_vals) ==  len(vals) - 1
            data[:, idx] = diff_vals
            # set labels
            if col == label_column:
                label_col_idx = idx

        # MaxMin scale input for NN
        if self.preprocessor:
            data = self.preprocessor.transform(data)
        else:
            self.preprocessor = MinMaxScaler() 
            self.preprocessor.fit(data)
            data = self.preprocessor.transform(data)

        # set train set and labels
        labels = data[1:len(data), label_col_idx]
        data = data[0:len(data)-1, :]
        return data, labels
    
    
    def __preprocess_data_v3(self, df, feature_columns, label_column, time_steps, preprocess=True):
        """
            1. Compute difference btw successive rows for each feature column
            2. Apply MaxMinScaler 
            3. Generate time step data
            3. Split data and labels by the give feature column
            
            labels are defined to be the next defined column value of 
            the each row
        """
        # set features
        df_data = pd.DataFrame()
        df_data['Timestamp'] = df['Timestamp'].values
        for idx, col in enumerate(feature_columns):
            # compute features
            vals = df[col].values.flatten()
            diff_vals = self.difference(vals)
            # validate features
            assert diff_vals[0] == vals[1] - vals[0]
            assert len(diff_vals) ==  len(vals) - 1
            # place features
            df_data[col] = diff_vals.astype('float32')
        df_data = df_data.drop(df_data.index[-1])

        # MaxMin scale input for NN
        if preprocess:
            if self.preprocessor:
                data = self.preprocessor.transform(df_data[feature_columns])
                df_data[feature_columns] = data
            else:
                self.preprocessor = MinMaxScaler() 
                self.preprocessor.fit(df_data[feature_columns])
                data = self.preprocessor.transform(df_data[feature_columns])
                df_data[feature_columns] = data

        if time_steps > 1:
            cols = []
            for i in range(time_steps):
                if i == 0:
                    col = df_data
                else:
                    col = df_data[feature_columns].shift(i)
                    new_names = {col: col+"_"+str(i) for col in df_data.columns}
                    col = col.rename(columns=new_names)
                cols.append(col)
            df_data = pd.concat(cols, axis=1)

        # set train set and labels
        labels = df_data[label_column].shift(-1)
        df_data = df_data.drop(df_data.index[-1])  # skip the last row since it has no diff
        df_data['label'] = labels  # skip the first since no prev value
        df_data = df_data.dropna()
        return df_data
    
    
    def __preprocess_data_v4(self, df, time_steps, preprocess=True):
        """
        for the paper https://arxiv.org/pdf/1410.1231v1.pdf
        """
        # move features columns to a new DF
        feature_columns = ['Close', 'Volume_(BTC)', 'Volume_(Currency)']
        prices = df['Close'].values.flatten().astype('float32')
        v_bid = df['Volume_(BTC)'].values.flatten().astype('float32')
        v_ask = df['Volume_(Currency)'].values.flatten().astype('float32')

        m = len(prices) - time_steps
        ts = np.empty((m, time_steps + 1))
        for i in range(m):
            ts[i, :time_steps] = prices[i:i + time_steps]
            ts[i, time_steps] = prices[i + time_steps] - prices[i + time_steps - 1]
        prices = ts

        # set train set and labels
        return prices, v_bid, v_ask
            

    def difference(self, dataset, interval=1):
        """ Remove trend for stationary data"""
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return pd.Series(diff)


def sample_entropy(time_series, sample_length, tolerance=None):
    """Calculate and return Sample Entropy of the given time series.
    Distance between two vectors defined as Euclidean distance and can
    be changed in future releases

    Args:
        time_series: Vector or string of the sample data
        sample_length: Number of sequential points of the time series
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))

    Returns:
        Vector containing Sample Entropy (float)

    References:
        [1] http://en.wikipedia.org/wiki/Sample_Entropy
        [2] http://physionet.incor.usp.br/physiotools/sampen/
        [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis
            of biological signals
    """
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    n = len(time_series)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((sample_length, 1))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((sample_length, 1))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = time_series[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(time_series[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]

    N = n * (n - 1) / 2
    B = np.vstack(([N], B[:sample_length - 1]))
    similarity_ratio = A / B
    se = - np.log(similarity_ratio)
    se = np.reshape(se, -1)
    return se