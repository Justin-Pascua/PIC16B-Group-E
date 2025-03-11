# Note: run pip install polygon-api-client
# get API key at https://polygon.io/stocks?utm_term=polygon%20api%20key&utm_campaign=Brand+-+ALL+(Conv+Value+tROAS)&utm_source=adwords&utm_medium=ppc&hsa_acc=4299129556&hsa_cam=14536485495&hsa_grp=132004734661&hsa_ad=591202818400&hsa_src=g&hsa_tgt=kwd-1443312992629&hsa_kw=polygon%20api%20key&hsa_mt=e&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=Cj0KCQjwm7q-BhDRARIsACD6-fXXpjDRkXM9S6_QG4Q-wOpYGAVtxv0rOzek563ebR5vHTy89qfGIvwaApzHEALw_wcB

import pandas as pd
from polygon import RESTClient
import requests
from datetime import datetime, timezone
import time

def timestamp_to_date(timestamp):
    """
    Convert a UNIX timestamp in milliseconds to a date string in the format YYYY-MM-DD.
    """
    
    # Convert to seconds and shift by a day to align with data
    epoch_secs = timestamp/1000 + 86400
    
    # Convert the timestamp to a datetime object
    dt_object = datetime.fromtimestamp(epoch_secs)
    
    # Convert to string of form YYYY-MM-DD
    date_string = dt_object.strftime('%Y-%m-%d')
    
    return date_string

def date_to_timestamp(date):
    """
    Convert a string of the form YYYY-MM-DD to the timestamp of that day at 05:00:00
    """
    
    # Parse the input date string and set the time to 05:00:00
    date_obj = datetime.strptime(date, '%Y-%m-%d').replace(hour=5, minute=0, second=0, microsecond=0)
    
    # Calculate the UNIX timestamp in seconds (timezone-naive, assuming UTC)
    timestamp_seconds = (date_obj - datetime(1970, 1, 1)).total_seconds()
    
    # Convert to milliseconds
    timestamp_milliseconds = int(timestamp_seconds * 1000)
    
    return timestamp_milliseconds
    
def exponential_smoothing(a, stock_data):
    """
    Applies exponential smoothing to each column (except timestamp and date) 
    of the dataframe
    0 < alpha < 1
    At alpha = 1, the smoothed data is equal to the initial data
    """
    stock_data[['Close', 'Volume', 'Open', 'High', 'Low']] = stock_data[['Close', 'Volume', 'Open', 'High', 'Low']].ewm(alpha = a, adjust = True).mean()
    return stock_data

class API_interface:
    def __init__(self, key):
        # initializes API key string 
        self.key = key
        self.client = RESTClient(key)

    def multiple_stock_table(self, date):
        """
        Returns a dataframe containing all info on all stocks at a single specified date
        Date is a string of the form YYYY-MM-DD
        Dataframe is sorted by trade volume
        """
        
        # get data from API
        data = self.client.get_grouped_daily_aggs(date)
    
        # read into dataframe
        df = pd.DataFrame({
            'Close' : [stock.close for stock in data],
            'Volume': [stock.volume for stock in data],
            'Open' : [stock.open for stock in data],
            'High' : [stock.high for stock in data],
            'Low' : [stock.low for stock in data], 
            'Vwap': [stock.vwap for stock in data],
            'Transactions': [stock.transactions for stock in data]},
            index = [stock.ticker for stock in data])
        
        # sort by trading volume
        df = df.sort_values('Volume', ascending = False)
        
        return df
    
    def single_stock_table(self, ticker, start_date, end_date):
        """
        Returns a dataframe containing all info on a given stock from start_date to end_date
        Dates are strings of the form YYYY-MM-DD
        """
        
        # get data from API
        data = self.client.get_aggs(ticker, 1, 'day', start_date, end_date)
        
        # read data into dataframe
        df = pd.DataFrame({
        'Timestamp' : [entry.timestamp for entry in data],
        'Date' : [timestamp_to_date(entry.timestamp) for entry in data],
        'Close' : [entry.close for entry in data],
        'Volume': [entry.volume for entry in data],
        'Open' : [entry.open for entry in data],
        'High' : [entry.high for entry in data],
        'Low' : [entry.low for entry in data], 
        'Vwap': [entry.vwap for entry in data],
        'Transactions': [entry.transactions for entry in data]}
        )
        
        return df
    
    def fetch_macd(self, stock_ticker, start_date, end_date, short_window = 12, long_window = 26, signal_window = 9):
        """
        Fetch MACD data from Polygon API for a given stock.
    
        If the MACD is higher than the signal line there is a upward trend and should think of buying 
        If the MACD is lower than the signal line there is a downwards trend and should think of selling
        
        start_date and end_date are strings of the form YYYY-MM-DD
        """
        
        # get data from API
        data = self.client.get_macd(stock_ticker, timespan = 'day', limit = 5000, adjusted = True, short_window = short_window, long_window = long_window, signal_window = signal_window, series_type = 'close', order = 'asc')
    
        # read data into dataframe
        df = pd.DataFrame({
        'Timestamp' : [entry.timestamp for entry in data.values],
        'Date' : [timestamp_to_date(entry.timestamp) for entry in data.values],   
        'MACD Value' : [entry.value for entry in data.values],
        'Signal' : [entry.signal for entry in data.values],
        })
        
        # filter by entries which are in the date range
        df = df[(df['Timestamp'] >= date_to_timestamp(start_date)) & (df['Timestamp'] <= date_to_timestamp(end_date))]
        
        return df
    
    def stock_proc(self, ticker, start_date, end_date, n):
        """
        Fetch stock data for a given ticker between start_date and end_date,
        and compute the Price Rate of Change (PROC) over the last 'n' days.
        Automatically fixes incorrect date order.
        """
        # Ensure correct date order
        start_date, end_date = min(start_date, end_date), max(start_date, end_date)
    
        # Fetch stock data from API
        data = self.client.get_aggs(ticker, 1, 'day', start_date, end_date)
        
        # Convert data into a DataFrame
        df = pd.DataFrame({
            'Close': [entry.close for entry in data],
            'Volume': [entry.volume for entry in data],
            'Open': [entry.open for entry in data],
            'High': [entry.high for entry in data],
            'Low': [entry.low for entry in data], 
            'Vwap': [entry.vwap for entry in data],
            'Transactions': [entry.transactions for entry in data]
        }, index=[timestamp_to_date(entry.timestamp) for entry in data])
    
        # Ensure data is sorted by date
        df.sort_index(inplace=True)
        
        # Compute PROC
        df["Close_n_days_ago"] = df["Close"].shift(n)  # Closing price 'n' days ago
        df["PROC"] = ((df["Close"] - df["Close_n_days_ago"]) / df["Close_n_days_ago"]) * 100
        
        # Get the latest row with valid PROC
        latest_data = df.iloc[-1][["Close", "Close_n_days_ago", "PROC"]]
        
        # Return the final DataFrame
        result = pd.DataFrame({
            "Ticker": [ticker],
            "Current Close": [latest_data["Close"]],
            f"Close {n} Days Ago": [latest_data["Close_n_days_ago"]],
            f"PROC ({n} days)": [latest_data["PROC"]]
        })
        
        return result
    
    def fetch_rsi(self, stock_ticker, start_date, end_date, timespan="day", window=14):
        """
        Fetches the Relative Strength Index (RSI) for a given stock ticker.
        RSI > 70 → Overbought (possible price decrease)
        RSI < 30 → Oversold (possible price increase)
        """
        
        # get data from API
        data = self.client.get_rsi(stock_ticker, window = window, timespan = 'day', limit = 5000, order = 'asc')
    
        # read data into dataframe
        df = pd.DataFrame({
        'Timestamp' : [entry.timestamp for entry in data.values],
        'Date' : [timestamp_to_date(entry.timestamp) for entry in data.values],
        'RSI Value' : [entry.value for entry in data.values]
        })
        
        # filter by entries which are in the date range
        df = df[(df['Timestamp'] >= date_to_timestamp(start_date)) & (df['Timestamp'] <= date_to_timestamp(end_date))]
    
        return df
    
    def calculate_williams_r(self, stock_data, window=14):
        """
        Calculates Williams %R for the given stock data.
        Above -20 → Sell signal
        Below -80 → Buy signal
        """
        
        # Calculate Williams %R using a rolling window
        stock_data["Williams %R"] = stock_data.apply(
            lambda row: (
                (max(stock_data.loc[row.name - window + 1: row.name, "High"]) - row["Close"]) /
                (max(stock_data.loc[row.name - window + 1: row.name, "High"]) - 
                 min(stock_data.loc[row.name - window + 1: row.name, "Low"]))
            ) * -100 if row.name >= window - 1 else None, axis=1
        )
    
        return stock_data
    
    def calculate_KDJ(self, stock_data, window_k_raw = 9, window_d = 3, window_k_smooth = 3):
        """
        Calculates the KDJ indicators 
        """
        
        # Calculate RSV (i.e. raw %K)
        stock_data['H'] = stock_data['High'].rolling(window = window_k_raw).max()
        stock_data['L'] = stock_data['Low'].rolling(window = window_k_raw).min()
    
        stock_data['RSV'] = (
            (stock_data['Close'] - stock_data['L']) /
            (stock_data['H'] - stock_data['L'])
        ) * 100
        
        # Calculate %K by taking a moving average of RSV
        stock_data['%K'] = stock_data['RSV'].rolling(window = window_k_smooth).mean()
        
        # Calculate %D by taking a moving average of %K
        stock_data['%D'] = stock_data['%K'].rolling(window = window_d).mean()
        
        # Calculate %J by 
        stock_data['%J'] = 3 * stock_data['%K'] - 2 * stock_data['%D']
        
        # Omit intermediate columns
        stock_data = stock_data.drop(columns = ['H', 'L', 'RSV'])
    
        return stock_data
    
    def get_all_features(self, ticker, start_date, end_date, alpha = 0.8, smoothing = True,
                     macd_short_window = 12, macd_long_window = 26, macd_signal_window = 9, 
                     rsi_timespan = 'day', rsi_window=14,
                     williams_r_window = 14, 
                     window_k_raw = 9, window_d = 3, window_k_smooth = 3):
        """
        Returns a data frame with all stock info and indicators
            :param ticker: The ticker symbol for the stock",
            :param start_date: A string of the form YYYY-MM-DD, earliest date to pull data from,
            :param end_date: A string of the form YYYY-MM-DD, latest date to pull data from,
            :param alpha: The parameter for exponential smoothing,
            :param smoothing: Specifies whether or not to apply exponential smoothing to raw data
            :param macd_short_window: The short window size used to compute MACD,
            :param macd_long_window: The long window size used to compute MACD,
            :param macd_signal_window: The window size used to calculate signal line,
            :param rsi_timespan: The size of underlying aggregate time window when computing RSI,
            :param rsi_window: The window size used to calculate the simple moving average for RSI,
            :param williams_r_window: The window size used to calculate the Williams %R,
            :param window_k_raw: The window size used to compute the raw %K,
            :param window_d: The window size used of the moving average used to compute %D,
            :param window_k_smooth: The window size of the moving average used to compute smooth %K,
        """
        
        # get close, vol, open, high, low, vwap, and transactions
        basic_data = self.single_stock_table(ticker, start_date, end_date)
        
        # exponentially smooth data
        if smoothing:
            basic_data = exponential_smoothing(alpha, basic_data)
        
        # get macd and signal
        macd_data = self.fetch_macd(ticker, start_date, end_date, macd_short_window, macd_long_window, macd_signal_window)
        
        # get rsi
        rsi_data = self.fetch_rsi(ticker, start_date, end_date, rsi_timespan, rsi_window)
        
        # merge basic and macd by inner joining along timestamps and dates
        df = pd.merge(basic_data, macd_data, how = 'inner', on = ['Timestamp', 'Date', 'Timestamp', 'Date'])
        
        # merge with rsi by inner joining along timestamps and dates
        df = pd.merge(df, rsi_data, how = 'inner', on = ['Timestamp', 'Date', 'Timestamp', 'Date'])
        
        # compute Williams%R and KDJ indicators
        df = self.calculate_williams_r(self.calculate_KDJ(df))
        
        # returns final dataframe excluding rows with N/A entries
        return df.dropna().reset_index(drop = True)
    
