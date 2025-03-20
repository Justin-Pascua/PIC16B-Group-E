import numpy as np
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers

import jax.numpy as jnp
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

import yahoo_interface
import math

# Used to keep values in [0,1]
def proj(x, a, b):
  """
  Projects a number x onto the interval [a,b]
  Values in [a,b] are projected onto themselves
  Values less than a are projected onto a
  Values greater than b are projected onto b
    :param x: number to be projected onto [a,b]
    :param a: left endpoint of interval [a,b]
    :param b: right endpoint of interval [a,b]
  """
  return max(min(x, b), a)

# Used to add noise to stock predictions
def noise():
    """
    Takes one sample of a normal distribution with mean 0, stdev = 0.33, projected onto [-1,1]
    """
    value = np.random.normal(0, 0.33, 1)
    return proj(value, -1,1 )

# function for creating training sequences to feed into model
def create_training_sequences(data, window_size = 60):
    """
      Returns a list of windows in data of length window_size
          and a list containing corresponding to the data value immediately proceeding a window

      :param data: a numpy array of shape (n, 1), e.g. closing_prices
      :param window_size: length of window
    """

    # initialize output lists
    X, y = [], []
    
    # take observations
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size, 0])    # closing prices in last (seq_length)-days
        y.append(data[i + window_size, 0])       # closing price in day after the x-sequence
    return np.array(X), np.array(y)

# function for creating training sets
def create_training_sets(tickers, start_date, end_date, chunk_size = 365, smoothing = True, alpha = 0.5):
    """
    Returns a training set and associated targets.    
        :param tickers: An array of strings representing stock tickers. This specifies which stocks to get observations from.
        :param start_date: A string of the form YYYY-MM-DD specifying the start of the date range
        :param end_date: A string of the form YYYY-MM-DD specifying the end of the date range
        :param chunk_size: An integer specifying the number of records in each chunk 
        :param smoothing: A boolean indicating whether or not to apply exponential smoothing when calling yahoo_interface.get_all_features()
        :param alpha: A float in [0,1], i.e. the exponential smoothing parameter, which is passed to yahoo_interface.get_all_features()
    """
    
    window_size = 60
    X_train = np.empty((0,60), dtype = 'float64')
    y_train = np.empty((0), dtype = 'float64')
    
    for ticker in tickers:
      df = yahoo_interface.get_all_features(ticker, start_date, end_date, smoothing = smoothing, alpha = alpha)
      for i in range(len(df)//chunk_size):
        # get chunk of dataframe
        df_chunk = df[i*chunk_size: min((i+1)*chunk_size, len(df))]
    
        # Put closing prices into numpy array
        closing_prices = df_chunk['Close'].values.reshape(-1, 1)
    
        # Normalize the data between [0,1]
        scaler = MinMaxScaler(feature_range = (0, 1))
        scaled_data = scaler.fit_transform(closing_prices)
        
        X, y = create_training_sequences(scaled_data, window_size)
        X_train = np.concatenate((X_train, X))
        y_train = np.concatenate((y_train, y))

    return X_train, y_train
    
# function for creating training/validation sets
def create_training_and_val_sets(tickers, start_date, end_date, chunk_size = 365, smoothing = True, alpha = 0.5, split = 0.7):
    """
    Returns a training and validation sets.    
        :param tickers: An array of strings representing stock tickers. This specifies which stocks to get observations from.
        :param start_date: A string of the form YYYY-MM-DD specifying the start of the date range
        :param end_date: A string of the form YYYY-MM-DD specifying the end of the date range
        :param chunk_size: An integer specifying the number of records in each chunk 
        :param smoothing: A boolean indicating whether or not to apply exponential smoothing when calling yahoo_interface.get_all_features()
        :param alpha: A float in [0,1], i.e. the exponential smoothing parameter, which is passed to yahoo_interface.get_all_features()
        :param split: A float in [0,1] indicating what portion of the data is to be used for training vs validation
    """
    
    window_size = 60
    X_train = np.empty((0,60), dtype = 'float64')
    y_train = np.empty((0), dtype = 'float64')
    
    X_val = np.empty((0,60), dtype = 'float64')
    y_val = np.empty((0), dtype = 'float64')
    
    for ticker in tickers:
      df = yahoo_interface.get_all_features(ticker, start_date, end_date, smoothing = smoothing, alpha = alpha)
      for i in range(len(df)//chunk_size):
        # get chunk of dataframe
        df_chunk = df[i*chunk_size: min((i+1)*chunk_size, len(df))]
    
        # Put closing prices into numpy array
        closing_prices = df_chunk['Close'].values.reshape(-1, 1)
    
        # Normalize the data between [0,1]
        scaler = MinMaxScaler(feature_range = (0, 1))
        scaled_data = scaler.fit_transform(closing_prices)
    
        train_size = int(len(df_chunk) * split)
        train, val = scaled_data[: train_size, :], scaled_data[train_size:, :]
        
        X, y = create_training_sequences(train, window_size)
        X_train = np.concatenate((X_train, X))
        y_train = np.concatenate((y_train, y))
        
        X, y = create_training_sequences(val, window_size)
        X_val = np.concatenate((X_val, X))
        y_val = np.concatenate((y_val, y))
    
    return X_train, y_train, X_val, y_val

# modified version of create_training_sequences. Only outputs X array, not y array
def create_sequences(data, window_size = 60):
    
    """
      Returns a list of windows in data of length window_size
          and a list containing corresponding to the data value immediately proceeding a window

      :param data: a numpy array of shape (n, 1), e.g. closing_prices
      :param window_size: length of window
    """

    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i: i + window_size, 0])    # closing prices in last (seq_length)-days
    return np.array(X)

# creates LSTM model
def create_LSTM(imported_weights = None, training_set = None):
    """
    Creates and returns an LSTM model trained in one of two ways:
        if imported_weights is specified, then weights are loaded in using model.load_weights()
        if training_set is specified, then model is trained using model.fit()
    If both ways are specified, then imported_weights takes precedence
    If none of the two ways are specified, then method calls create_training_sets
    to create a training set.
        :param imported_weights: a string ending in '.weights.h5', indicating the name of a weights file which is compatible with the architecture of the LSTM model below
        :param training_set: a tuple of the form (X_train, y_train), where X_train is a numpy array of size (n, 60) and y_train is of size (n,), where n is the number of records in the training data
    """
    
    # create model
    model = keras.Sequential([
        layers.LSTM(32, return_sequences = True, input_shape = (60, 1)),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation = 'relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    # compile model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # if weights specified, import and return
    if imported_weights != None:
        model.load_weights(imported_weights)
    elif training_set != None:
        model.fit(training_set[0], training_set[1], epochs = 50)
    else:
        tickers = ['AAPL', 'COST', 'CVX', 'WM', 'LLY']
        start_date = '2000-01-01' 
        end_date = '2025-03-10'
        X_train, y_train = create_training_sets(tickers, start_date, end_date, 
                                                chunk_size = 365, smoothing = True, alpha = 0.5)
        model.fit(X_train, y_train, epochs = 50)
        
    return model
 
class LSTM_wrapper:
    def __init__(self, imported_weights = None, training_set = None):
        self.model = create_LSTM(imported_weights, training_set)
        
    def visualize_performance(self, tickers, start_date, end_date):
        scaler = MinMaxScaler(feature_range = (0, 1))
        
        plt.figure(figsize = (20,10))
        for i in range(6):
          # reading data
          df = yahoo_interface.get_all_features(f'{tickers[i]}', start_date, end_date, smoothing = False)
          closing_prices = df['Close'].values.reshape(-1, 1)
          scaled_data = scaler.fit_transform(closing_prices)
        
          # formatting data to be fed into model
          window_size = 60
          X_full = create_sequences(scaled_data, window_size)
        
          # getting predictions
          full_predict = self.model.predict(X_full)
          full_predict = scaler.inverse_transform(full_predict)
          mse = mean_squared_error(closing_prices[59:], full_predict)
        
          # Plotting real vs predicted prices
          known_plot = np.empty((closing_prices.shape[0] + 1, 1))
          known_plot[:, :] = np.nan
          known_plot[:closing_prices.shape[0], :] = closing_prices
        
          prediction_plot = np.empty((closing_prices.shape[0] + 1, 1))
          prediction_plot[:, :] = np.nan
          prediction_plot[window_size: closing_prices.shape[0] + 1, :] = full_predict
        
          # set up subplots
          plt.subplot(2,3,i+1)
          plt.title(f'{tickers[i]}')
          plt.plot(known_plot, color = 'blue', label = 'Actual Price')
          plt.plot(prediction_plot, color = 'orange', label = f'Predicted Price (MSE = {mse:.4f})')
          plt.legend()
        
        plt.show()
        
    def forecast(self, stock_data, days_forward = 60):
        """
        Generates predictions for a stock given 60 days of historical data
        Returns an appended version of scaled_data with predictions appearing at the end of the array
            :param stock_data: a dataframe containing a column titled 'Close', with at least 60 rows
            :param days_forward: the number of days in the future we want to forecast
        """
        scaler = MinMaxScaler(feature_range = (0, 1))
        scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
        
        # initialize array to store future values
        forecast_arr = scaled_data
        
        # get predictions
        for i in range(days_forward):
            # get last 60 days in array
            seq = create_sequences(forecast_arr[-60:], 60)
        
            # predict next day using model
            one_predict = self.model.predict(seq, verbose = 0)
        
            # add noise to prediction
            updated_value = np.array(one_predict[0][0].item()*(1 + 0.1 * noise()))
            one_predict[0][0] = updated_value.item()
        
            # add prediction to array
            forecast_arr = np.concatenate((forecast_arr, one_predict))
        
        # reverse data scaling
        forecast_arr = scaler.inverse_transform(forecast_arr)
        
        return forecast_arr
        
    def forecast_ensemble(self, stock_data, days_forward = 60, size = 10):
        """
        Calls forecast() several times and returns a list of forecast arrays
            :param stock_data: a dataframe containing a column titled 'Close', with at least 60 rows
            :param days_forward: the number of days in the future we want to forecast
            :param size: size of output list, i.e. number of times forecast() is called
        """
        forecast_list = []
        for i in range(size):
            forecast_list.append(self.forecast(self, stock_data, days_forward))

        return forecast_list