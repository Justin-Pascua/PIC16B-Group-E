import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import layers

from datetime import datetime, timezone
from datetime import datetime, timedelta

import jax.numpy as jnp
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import yahoo_interface

# used to create training sequences
def create_training_sequences(ticker, start_date, end_date, chunk_size = 365, smoothing = True, alpha = 0.5):
    """
    Returns a training set consisting of data from one stock ticker
        :param ticker: A string indicating a single stock ticker
        :param start_date: A string of the form YYYY-MM-DD specifying the start of the date range
        :param end_date: A string of the form YYYY-MM-DD specifying the end of the date range
        :param chunk_size: An integer specifying the number of records in each chunk 
        :param smoothing: A boolean indicating whether or not to apply exponential smoothing when calling yahoo_interface.get_all_features()
        :param alpha: A float in [0,1], i.e. the exponential smoothing parameter, which is passed to yahoo_interface.get_all_features()
    """
    
    # identify features used in model
    features = ['Close', 'High', 'Low', 'MACD Value', 'Signal', 'RSI Value', '%K', '%D', '%J', 'Williams %R']
    
    # get dataframe and targets
    df = yahoo_interface.get_all_features(ticker, start_date, end_date, smoothing = smoothing, alpha = alpha)
    df['Target'] = np.sign(df['Close'].shift(-1) - df['Close'])
    df = df.dropna()

    # define scaler and label encoder
    scaler = MinMaxScaler(feature_range = (0, 1))
    le = LabelEncoder()

    # initialize outputs
    X = []
    y = []

    # parse through chunks
    for i in range(len(df)//chunk_size + 1):
        # isolate a chunk
        df_chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size].copy()
        for feature in features:
            data_chunk = df_chunk[feature]  # Get features in chunk
            scaled_chunk = scaler.fit_transform(data_chunk.values.reshape(-1, 1)).flatten() # Normalize selected feature chunk
            df_chunk.loc[data_chunk.index, feature] = scaled_chunk  # Assign to dataframe

        # create sequences
        for start in range(0, len(df_chunk) - 14):
            end = start + 14
            window = df_chunk[features].iloc[start: end].to_numpy()
            target = df_chunk['Target'].iloc[end]
            X.append(window)
            y.append(target)

    X = np.array(X, dtype = np.float32)
    y = le.fit_transform(y)

    return X, y

# same as create_training_sequences, but accepts multiple tickers
def create_training_sets(tickers, start_date, end_date, chunk_size = 365, smoothing = True, alpha = 0.5):
    """
    Returns a training set consisting of data from multiple stock tickers
        :param tickers: An array of strings representing stock tickers
        :param start_date: A string of the form YYYY-MM-DD specifying the start of the date range
        :param end_date: A string of the form YYYY-MM-DD specifying the end of the date range
        :param chunk_size: An integer specifying the number of records in each chunk 
        :param smoothing: A boolean indicating whether or not to apply exponential smoothing when calling yahoo_interface.get_all_features()
        :param alpha: A float in [0,1], i.e. the exponential smoothing parameter, which is passed to yahoo_interface.get_all_features()
    """
    
    # initialize output arrays
    X_train = np.empty((0,14,10), dtype = 'float64')
    y_train = np.empty((0), dtype = 'float64')
    
    # add to training set by calling create_training_sequences
    for ticker in tickers:
      X, y = create_training_sequences(ticker, '2000-01-01', '2025-01-01', alpha = 0.8)
      X_train = np.concatenate((X_train, X), axis = 0)
      y_train = np.concatenate((y_train, y), axis = 0)
    
    return X_train, y_train

# modified version of create_training_sequences which does not output target labels
def create_test_sequences(ticker, start_date, end_date, chunk_size = 365, smoothing = True, alpha = 0.5):
    """
    Returns a test set consisting of data from one stock ticker
        :param ticker: A string indicating a single stock ticker
        :param start_date: A string of the form YYYY-MM-DD specifying the start of the date range
        :param end_date: A string of the form YYYY-MM-DD specifying the end of the date range
        :param chunk_size: An integer specifying the number of records in each chunk 
        :param smoothing: A boolean indicating whether or not to apply exponential smoothing when calling yahoo_interface.get_all_features()
        :param alpha: A float in [0,1], i.e. the exponential smoothing parameter, which is passed to yahoo_interface.get_all_features()
    """
    
    # identify features used in model
    features = ['Close', 'High', 'Low', 'MACD Value', 'Signal', 'RSI Value', '%K', '%D', '%J', 'Williams %R']
    
    # get dataframe and targets
    df = yahoo_interface.get_all_features(ticker, start_date, end_date, smoothing = smoothing, alpha = alpha)
    df['Target'] = np.sign(df['Close'].shift(-1) - df['Close'])
    df = df.dropna()

    # define scaler and label encoder
    scaler = MinMaxScaler(feature_range = (0, 1))
    le = LabelEncoder()

    # initialize output
    X = []

    # parse through chunks
    for i in range(len(df)//chunk_size + 1):
        # isolate a chunk
        df_chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size].copy()
        for feature in features:
            data_chunk = df_chunk[feature]  # Get features in chunk
            scaled_chunk = scaler.fit_transform(data_chunk.values.reshape(-1, 1)).flatten() # Normalize selected feature chunk
            df_chunk.loc[data_chunk.index, feature] = scaled_chunk  # Assign to dataframe

        # create sequences
        for start in range(0, len(df_chunk) - 14):
            end = start + 14
            window = df_chunk[features].iloc[start: end].to_numpy()
            target = df_chunk['Target'].iloc[end]
            X.append(window)

    X = np.array(X, dtype = np.float32)

    return X

def create_CNN(imported_weights = None, training_set = None):
    """
    Creates and returns a CNN model trained in one of two ways:
        if imported_weights is specified, then weights are loaded in using model.load_weights()
        if training_set is specified, then model is trained using model.fit()
    If both ways are specified, then imported_weights takes precedence
    If none of the two ways are specified, then method calls create_training_sets
    to create a training set.
        :param imported_weights: a string ending in '.weights.h5', indicating the name of a weights file which is compatible with the architecture of the LSTM model below
        :param training_set: a tuple of the form (X_train, y_train), where X_train is a numpy array of size (n, 14, 10) and y_train is of size (n,), where n is the number of records in the training data
    """
    
    # create model
    model = keras.Sequential([
        layers.Input((14, 10, 1)),
        layers.Conv2D(32, (3, 3), padding = 'same', activation='relu'),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding = 'same', activation='relu'),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding = 'same',activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)
    ])
    
    # compile model
    model.compile(optimizer = 'adam',
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])
    
    # if weights specified, import
    if imported_weights != None:
        model.load_weights(imported_weights)
    # if training set specified, fit to training data
    elif training_set != None:
        model.fit(training_set[0], training_set[1], epochs = 50)
    # otherwise, train with the following defualt data set
    else:
        tickers = ['AAPL', 'COST', 'CVX', 'WM', 'LLY']
        start_date = '2000-01-01' 
        end_date = '2025-03-10'
        X_train, y_train = create_training_sets(tickers, start_date, end_date, 
                                                chunk_size = 365, smoothing = True, alpha = 0.5)
        model.fit(X_train, y_train, epochs = 50)
        
    return model

class CNN_Wrapper:
    def __init__(self, imported_weights = None, training_set = None):
        self.model = create_CNN(imported_weights, training_set)
        
    def evaluate(self, ticker, start_date, end_date, smoothing = True, alpha = 0.5):
        """
        Tests model accuracy against stock data on a specified ticker over a specified range
            :param ticker: A string indicating a single stock ticker
            :param start_date: A string of the form YYYY-MM-DD specifying the start of the date range
            :param end_date: A string of the form YYYY-MM-DD specifying the end of the date range
            :param chunk_size: An integer specifying the number of records in each chunk 
            :param smoothing: A boolean indicating whether or not to apply exponential smoothing when calling yahoo_interface.get_all_features()
            :param alpha: A float in [0,1], i.e. the exponential smoothing parameter, which is passed to yahoo_interface.get_all_features()
        """
        
        # create test set
        X_test, y_test = create_training_sequences(ticker, start_date, end_date, smoothing = smoothing, alpha = alpha)
        
        # evaluate model
        self.model.evaluate(X_test, y_test)

    def predict(self, ticker):
        """
        Predicts whether or not a stock's current closing price will rise or fall tomorrow
            :param ticker: a string specifying a stock ticker
        """
        
        # identify classes
        classes = ['Rise', 'Fall']
        
        # get today's timestamp
        current_timestamp = datetime.now().timestamp()*1000
        
        # get timestamp for 5 months back (used to compute features)
        past_timestamp = current_timestamp - 86400 * 31 * 5 * 1000
        
        # convert timestamps to strings of form YYYY-MM-DD
        current_date = yahoo_interface.timestamp_to_date(current_timestamp)
        past_date = yahoo_interface.timestamp_to_date(past_timestamp)
        
        # get sequences to be fed into model
        X_test = create_test_sequences(ticker, past_date, current_date, smoothing = False)
        
        # get predictions
        y_pred = self.model.predict(X_test)
        
        # apply argmax 
        labels = y_pred.argmax(axis = 1)
        
        # output prediction
        return classes[labels.item(-1)]