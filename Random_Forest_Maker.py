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

import sys

# Get the absolute path to the directory containing the .py file
module_path = os.path.abspath(os.path.join('..')) # Use relative or absolute path. '..' means one level up.

if module_path not in sys.path:
    sys.path.append(module_path)
    
import yahoo_interface

import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

def create_random_forest(training_set = None, 
                           n_estimators=100, 
                           max_depth=None, 
                           min_samples_split=2, 
                           min_samples_leaf=1, 
                           random_state=42):
    """
    Creates, trains, and returns a RandomForestRegressor model.
        :param training_set: a tuple of the form (X_train, y_train), where X_train is a numpy array of size (n, 60) and y_train is of size (n,), where n is the number of records in the training data
        :param n_estimators: number of trees in forest
        :param max_depth: max depth of a tree
        :param min_samples_split: minimum nnumber of samples required to split an internal node
        :param min_samples_leaf: minimum number of samples required to be at a leaf node
        :param random_state: parameter controling randomness of bootstrapping and sampling of features
    """
    
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=random_state)
    if training_set != None:
        model.fit(training_set[0], training_set[1])
    else:
        tickers = ['AAPL', 'COST', 'CVX', 'WM', 'LLY']
        start_date = '2000-01-01' 
        end_date = '2025-03-10'
        X_train, y_train = create_training_sets(tickers, start_date, end_date, 
                                                chunk_size = 365, smoothing = True, alpha = 0.5)
        model.fit(X_train, y_train)
        
    return model

class Random_Forest_wrapper:
    def __init__(self, training_set = None):
        """
        Initializes object using create_random_forest
            :param training_set: a tuple of the form (X_train, y_train), where X_train is a numpy array of size (n, 60) and y_train is of size (n,), where n is the number of records in the training data
        """
        self.model = create_random_forest(training_set = training_set)
   
    def mse_evaluation(self, X_val, y_val):
        """
        Makes predictions and evaluates the model's performance.
            :param X_val: a numpy array of shape (n,60) representing a validation data set
            :param y_val: a numpy array of shape (n,) representing the labels of the validation set
        """
        
        y_pred = self.model.predict(X_val)
    
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
    
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R-squared (R2): {r2}")
    
        return y_pred
    
    def scatter_plot_evaluation(self, y_val, y_pred):
        """
        Creates a scatter plot of actual vs. predicted values.
        """
        plt.scatter(y_val, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values (Random Forest Regression)")
        plt.show()
        
    def line_plot_evaluation(self, tickers, start_date, end_date):
        """
        Creates a line plot comparing actual vs predicted prices against multiple stocks
            :param tickers: a list of strings containing 6 stock tickers
            :param start_date: A string of the form YYYY-MM-DD specifying the start of the date range
            :param end_date: A string of the form YYYY-MM-DD specifying the end of the date range
        """
        scaler = MinMaxScaler(feature_range = (0, 1))
        
        plt.figure(figsize = (15,10))
        for i in range(6):
            # reading data
            df = yahoo_interface.get_all_features(f'{tickers[i]}', '2024-01-01', '2025-01-01', smoothing = False)
            closing_prices = df['Close'].values
            closing_prices = closing_prices[:,np.newaxis]
            scaled_data = scaler.fit_transform(closing_prices)
        
            # formatting data to be fed into model
            window_size = 60
            X_full = create_sequences(scaled_data, window_size)
        
            # getting predictions
            full_predict = self.model.predict(X_full)
            full_predict = full_predict[:,np.newaxis]
            full_predict = scaler.inverse_transform(full_predict)
        
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
            plt.plot(prediction_plot, color = 'orange', label = 'Predicted Price')
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
            seq = create_sequences(forecast_arr[-60:], 60)
            one_predict = self.model.predict(seq)
            one_predict[0] = one_predict[0]*(1 + 0.1 * noise())
            one_predict = one_predict[:,np.newaxis]
            forecast_arr = np.concatenate((forecast_arr, one_predict))
        
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
            forecast_list.append(self.forecast(stock_data = stock_data, days_forward = days_forward))

        return forecast_list
    
    