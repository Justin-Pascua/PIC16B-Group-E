# Note: run pip install yfinance
import sys

from datetime import datetime, timezone
from datetime import datetime, timedelta
import time
import yfinance as yf
import pandas as pd

# Function to convert UNIX timestamp (in milliseconds) to date string (YYYY-MM-DD)
def timestamp_to_date(timestamp):
    """
    Convert a UNIX timestamp in milliseconds to a date string in the format YYYY-MM-DD.
    """
    epoch_secs = timestamp / 1000  # Convert milliseconds to seconds
    dt_object = datetime.fromtimestamp(epoch_secs, tz=timezone.utc)  # Ensure UTC timezone
    date_string = dt_object.strftime('%Y-%m-%d')
    
    return date_string

# Function to convert date string (YYYY-MM-DD) to UNIX timestamp in milliseconds
def date_to_timestamp(date):
    """
    Convert a string of the form YYYY-MM-DD to the timestamp of that day at 05:00:00 UTC.
    """
    date_obj = datetime.strptime(date, '%Y-%m-%d').replace(hour=5, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    timestamp_milliseconds = int(date_obj.timestamp() * 1000)  # Convert to milliseconds
    
    return timestamp_milliseconds

def flatten_columns(df):
    """Flattens MultiIndex column names by taking top-most name."""
    df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    df.index.name = None
    return df

def exponential_smoothing(a, stock_data):
    """
    Applies exponential smoothing to each column (except timestamp and date) 
    of the dataframe
    0 < alpha < 1
    At alpha = 1, the smoothed data is equal to the initial data
    """
    stock_data[['Close', 'Volume', 'Open', 'High', 'Low']] = stock_data[['Close', 'Volume', 'Open', 'High', 'Low']].ewm(alpha = a, adjust = True).mean()
    return stock_data

def single_stock_table(ticker, start_date, end_date):
    """
    Returns a dataframe containing all available stock data from Yahoo Finance 
    for a given stock between start_date and end_date.
    Dates are in the format YYYY-MM-DD.
    """

    # Fetch stock data
    df = yf.download(ticker, start=start_date, end=end_date)

    # Reset index to get the Date as a column
    df.reset_index(inplace=True)

    # Rename columns to match the original function (except VWAP)
    df.rename(columns={
        'Date': 'Date',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }, inplace=True)
    
    # Convert the Date column to timestamps (milliseconds)
    df['Timestamp'] = df['Date'].astype('str').apply(date_to_timestamp)  # Convert to milliseconds
    
    # Rearrange columns (excluding VWAP)
    # Add this line inside the function to compute VWAP before rearranging columns
    df['Vwap'] = (df['High'] + df['Low'] + df['Close']) / 3
    df = df[['Timestamp', 'Date', 'Close', 'Volume', 'Open', 'High', 'Low', 'Vwap']]

    df = flatten_columns(df)
    
    return df
    
def multiple_stock_table(date):
    """
    Returns a dataframe containing stock info for multiple specified tickers on a given date.
    Date is a string in the format YYYY-MM-DD.
    Dataframe is sorted by trade volume.
    """
    
    # Define a list of commonly traded stocks (can be expanded)
    ticker_list = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA", "NVDA", "META", "NFLX"]
    
    # Convert date format for Yahoo Finance
    formatted_date = datetime.strptime(date, '%Y-%m-%d')
    next_day = formatted_date + timedelta(days=1)  # yfinance requires an end date
    
    # Fetch stock data
    data = yf.download(ticker_list, start=formatted_date.strftime('%Y-%m-%d'), 
                       end=next_day.strftime('%Y-%m-%d'), interval="1d")
    
    # Extract relevant fields
    df = pd.DataFrame({
        "Close": data["Close"].iloc[0],
        "Volume": data["Volume"].iloc[0],
        "Open": data["Open"].iloc[0],
        "High": data["High"].iloc[0],
        "Low": data["Low"].iloc[0]
    })
    
    # Sort by trading volume
    df = df.sort_values("Volume", ascending=False)
    
    df = flatten_columns(df)
        
    return df
    
def fetch_macd(stock_ticker, start_date, end_date, short_window=12, long_window=26, signal_window=9):
    """
    Fetch MACD data for a given stock using yfinance.

    - If MACD > Signal Line → Upward trend (potential buy signal)
    - If MACD < Signal Line → Downward trend (potential sell signal)
    
    Parameters:
    - stock_ticker: Stock symbol (e.g., "AAPL")
    - start_date, end_date: Date range (YYYY-MM-DD)
    - short_window: Short-term EMA period (default=12)
    - long_window: Long-term EMA period (default=26)
    - signal_window: Signal line EMA period (default=9)
    """

    # Fetch stock data from Yahoo Finance
    df = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")

    # Compute short-term and long-term exponential moving averages (EMA)
    df['Short_EMA'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['Long_EMA'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    
    # Calculate MACD Line
    df['MACD Value'] = df['Short_EMA'] - df['Long_EMA']

    # Calculate Signal Line (9-day EMA of MACD)
    df['Signal'] = df['MACD Value'].ewm(span=signal_window, adjust=False).mean()
    
    # Convert Date index to a column and format
    df.reset_index(inplace=True)
        
    # Add timestamp column
    df['Timestamp'] = df['Date'].astype(str).apply(date_to_timestamp)
        
    # Select relevant columns
    df = df[['Timestamp', 'Date', 'MACD Value', 'Signal']]
    
    df = df.iloc[long_window - 1:]
        
    df = flatten_columns(df)
        
    return df
    
def stock_proc(ticker, start_date, end_date, n):
    """
    Fetch stock data for a given ticker between start_date and end_date,
    and compute the Price Rate of Change (PROC) over the last 'n' days.
    Automatically fixes incorrect date order.

    Parameters:
    - ticker: Stock symbol (e.g., "AAPL")
    - start_date, end_date: Date range (YYYY-MM-DD)
    - n: Number of days for PROC calculation
    
    Returns:
    - A DataFrame with the latest available PROC value
    """
    
    start_date, end_date = min(start_date, end_date), max(start_date, end_date)

    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    close_cols = [col for col in df.columns if "Close" in col]
        
    close_col = close_cols[0]  
    
    df["Close"] = df[close_col].astype(float)
    
    df["Close_n_days_ago"] = df["Close"].shift(n).astype(float)  
    
    df["PROC"] = ((df["Close"] - df["Close_n_days_ago"]) / df["Close_n_days_ago"]) * 100
    
    df["PROC"] = df["PROC"].replace([float("inf"), -float("inf")], None)
    df["PROC"] = df["PROC"].fillna(0)
    
    if not df.empty:
        latest_data = df.iloc[-1][["Close", "Close_n_days_ago", "PROC"]].to_dict()
    else:
        latest_data = {"Close": None, "Close_n_days_ago": None, "PROC": None}
    
    result = pd.DataFrame({
        "Ticker": [ticker],
        "Current Close": [latest_data["Close"]],
        f"Close {n} Days Ago": [latest_data["Close_n_days_ago"]],
        f"PROC ({n} days)": [latest_data["PROC"]]
    })
    
    return result

def fetch_rsi(stock_ticker, start_date, end_date, window=14):
    """
    Fetches the Relative Strength Index (RSI) for a given stock ticker using yfinance.
    
    - RSI > 70 → Overbought (potential price drop)
    - RSI < 30 → Oversold (potential price increase)
    
    Parameters:
    - stock_ticker: Ticker symbol (e.g., "AAPL")
    - start_date, end_date: Date range (YYYY-MM-DD)
    - window: Period for RSI calculation (default = 14)

    Returns:
    - DataFrame with RSI values
    """

    # Fetch stock data
    df = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")

    # Calculate price change
    df["Price Change"] = df["Close"].diff()

    # Calculate gains and losses
    df["Gain"] = df["Price Change"].apply(lambda x: x if x > 0 else 0)
    df["Loss"] = df["Price Change"].apply(lambda x: -x if x < 0 else 0)

    # Calculate rolling average gains and losses
    avg_gain = df["Gain"].rolling(window=window, min_periods=1).mean()
    avg_loss = df["Loss"].rolling(window=window, min_periods=1).mean()

    # Compute Relative Strength (RS)
    rs = avg_gain / avg_loss
    df["RSI Value"] = 100 - (100 / (1 + rs))

    # Convert Date index to a column and format
    df.reset_index(inplace=True)
        
    # Add timestamp column
    df['Timestamp'] = df['Date'].astype(str).apply(date_to_timestamp)
        
    # Select relevant columns
    df = df[["Timestamp", "Date", "RSI Value"]]
        
    df = flatten_columns(df)
    df = df.iloc[window - 1:]
    
    return df
    
def calculate_williams_r(stock_data, window=14):
    """
    Calculates Williams %R for the given stock data.
    
    - Above -20 → Overbought (Sell signal)
    - Below -80 → Oversold (Buy signal)
    
    Parameters:
    - stock_data: DataFrame containing "High", "Low", and "Close" columns.
    - window: The period over which Williams %R is calculated (default = 14 days).
        
    Returns:
    - DataFrame with an additional "Williams %R" column.
    """
    
    # Compute the highest high and lowest low over the window period
    highest_high = stock_data["High"].rolling(window=window, min_periods=1).max()
    lowest_low = stock_data["Low"].rolling(window=window, min_periods=1).min()
    
    # Calculate Williams %R
    stock_data["Williams %R"] = ((highest_high - stock_data["Close"]) / 
                                 (highest_high - lowest_low)) * -100

    return stock_data
    
def calculate_KDJ(stock_data, window_k_raw=9, window_d=3, window_k_smooth=3):
    """
    Calculates the KDJ indicators for a given stock dataset.

    - %K = Smoothed RSV (Relative Strength Value)
    - %D = Moving Average of %K
    - %J = 3 * %K - 2 * %D (momentum signal)

    Parameters:
    - stock_data: DataFrame from `single_stock_table()`
    - window_k_raw: Period for RSV calculation (default = 9 days)
    - window_d: Period for %D smoothing (default = 3 days)
    - window_k_smooth: Period for %K smoothing (default = 3 days)

    Returns:
    - DataFrame with %K, %D, and %J indicators added.
    """

    # Flatten column headers if they are multi-indexed
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns]

    # Ensure the dataset contains necessary columns (matching single_stock_table())
    required_columns = {"High", "Low", "Close"}
    if not required_columns.issubset(stock_data.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(stock_data.columns)}. Available columns: {list(stock_data.columns)}")

    # Calculate RSV (Raw Stochastic Value)
    stock_data["H"] = stock_data["High"].rolling(window=window_k_raw, min_periods=1).max()
    stock_data["L"] = stock_data["Low"].rolling(window=window_k_raw, min_periods=1).min()

    # Prevent division by zero (if H == L, replace with 1 to avoid NaN)
    stock_data["RSV"] = ((stock_data["Close"] - stock_data["L"]) /
                         (stock_data["H"] - stock_data["L"]).replace(0, 1)) * 100

    # Calculate %K as a moving average of RSV
    stock_data["%K"] = stock_data["RSV"].rolling(window=window_k_smooth, min_periods=1).mean()

    # Calculate %D as a moving average of %K
    stock_data["%D"] = stock_data["%K"].rolling(window=window_d, min_periods=1).mean()

    # Calculate %J as a momentum signal
    stock_data["%J"] = 3 * stock_data["%K"] - 2 * stock_data["%D"]

    # Drop intermediate columns
    stock_data.drop(columns=["H", "L", "RSV"], inplace=True)

    return stock_data

def get_all_features(ticker, start_date, end_date, alpha = 0.8, smoothing = True,
                     macd_short_window=12, macd_long_window=26, macd_signal_window=9, 
                     rsi_window=14, 
                     williams_r_window=14, 
                     window_k_raw=9, window_d=3, window_k_smooth=3):
    """
    Returns a DataFrame with all stock indicators:
    - MACD (Moving Average Convergence Divergence)
    - RSI (Relative Strength Index)
    - Williams %R (Momentum indicator)
    - KDJ (Stock trend analysis)
    """
    
    basic_data = single_stock_table(ticker, start_date, end_date)
    macd_data = fetch_macd(ticker, start_date, end_date, 
                           macd_short_window, macd_long_window, macd_signal_window)
    rsi_data = fetch_rsi(ticker, start_date, end_date, rsi_window)
    
    # exponentially smooth data
    if smoothing:
        basic_data = exponential_smoothing(alpha, basic_data)

    #basic_data.rename(columns={"Date_": "Date"}, inplace=True)
    #macd_data.rename(columns={"Timestamp_": "Date"}, inplace=True)
    #rsi_data.rename(columns={"Timestamp_": "Date"}, inplace=True)

    df = pd.merge(basic_data, macd_data, how="inner", on=["Date", "Timestamp"])
    df = pd.merge(df, rsi_data, how="inner", on=["Date", "Timestamp"])

    high_col = [col for col in df.columns if "High" in col][0]
    low_col = [col for col in df.columns if "Low" in col][0]
    close_col = [col for col in df.columns if "Close" in col][0]
    
    df.rename(columns={high_col: "High", low_col: "Low", close_col: "Close"}, inplace=True)

    df = calculate_williams_r(df, williams_r_window)
    
    df = calculate_KDJ(df, window_k_raw, window_d, window_k_smooth)
    
    return df.dropna().reset_index(drop=True)