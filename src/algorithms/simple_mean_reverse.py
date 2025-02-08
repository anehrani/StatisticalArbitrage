
import numpy as np
import pandas as pd
import statsmodels.api as sm



def generate_trading_signals(cointegrated_series, params: dict):
    """
    Generate trading signals based on the cointegrated time series.
    
    Args:
        cointegrated_series (pandas.Series): The cointegrated time series.
        lower_threshold (float, optional): Lower threshold for buy signals. Defaults to -2.
        upper_threshold (float, optional): Upper threshold for sell signals. Defaults to 2.
        smoothing_window (int, optional): Window size for smoothing the series. Defaults to 10.
    
    Returns:
        list: A list of dictionaries containing the trading signals.
    """
    # Smooth the cointegrated series to reduce noise
    smoothed_series = cointegrated_series[ 'Cointegrated Series'].rolling(window=params["smoothing_window"]).mean()
    mean_smoothed_series = np.mean(smoothed_series)
    variance_smoothed_series = np.var(smoothed_series)
    # Initialize signals list
    signals = []
    
    # Loop through the smoothed series
    for time, value in smoothed_series.items():
        if np.isnan(value):
            continue
        if value < params["lower_threshold"]:
            # Generate a buy signal
            signals.append({
                "direction": "long",
                "quantity": 1.0,  # Adjust quantity based on your strategy
                "time": time.isoformat()
            })
        elif value > params["upper_threshold"]:
            # Generate a sell signal
            signals.append({
                "direction": "short",
                "quantity": 1.0,  # Adjust quantity based on your strategy
                "time": time.isoformat()
            })
    
    return signals

