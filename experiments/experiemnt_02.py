

# Import pandas
import pandas as pd
import yfinance as yf
import datetime
import time
import ccxt
from datetime import date
import numpy as np
from datetime import time

import statsmodels.tsa.stattools as ts
import statsmodels.api as sm


import matplotlib.pyplot as plt



# import you







# Read data from exchange api, and save to dataframe
# params = {'pair': symbol, 'contractType': contractType, 'interval': time_interval, 'limit': 96,
#                   'startTime': start_time_since, 'endTime': loop_end_time}
# df = exchange.fapiPublicGetContinuousKlines(params=params)

class DataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance()


    # Store the list in a Dataframe
    def get_symbols( self):
        # get contractor symbols from biance exchange
        exinfo = self.exchange.fapiPublicGetExchangeInfo()['symbols']
        df = pd.DataFrame(exinfo)
        symbol_list = df['symbol'].tolist()
        return symbol_list

    def get_data( self, request:dict):
        # request should be like 
        # params = {'pair': symbol, 'contractType': contractType, 'interval': time_interval, 'limit': 96,
        #                   'startTime': start_time_since, 'endTime': loop_end_time}
        df = self.exchange.fapiPublicGetContinuousKlines(params=request)
        return df

class DataFetcher_:
    def __init__(self):
        self.exchange = ccxt.binance()
        
    def get_symbols(self):
        exinfo = self.exchange.fapiPublicGetExchangeInfo()['symbols']
        df = pd.DataFrame(exinfo)
        symbol_list = df['symbol'].tolist()
        return symbol_list
    
    def get_data(self, symbol, contract_type='PERPETUAL', interval='1h', 
                 start_date=None, end_date=None):
        if start_date is not None:
            start_date = int(pd.to_datetime(start_date).timestamp() * 1000)
        if end_date is not None:
            end_date = int(pd.to_datetime(end_date).timestamp() * 1000)
            
        params = {
            'pair': symbol,
            'contractType': contract_type,
            'interval': interval,
            'startTime': start_date,
            'endTime': end_date
        }
        
        data = self.exchange.fapiPublicGetContinuousKlines(params=params)
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        return df


def adf_cointegration_test( ts1, ts2, percent="10%"):
   """
   :param ts1:
   :param ts2:
   :param symbol1:
   :param symbol2:
   :return:
   """
   # Perform ADF test on the closing prices of fetched data
   result = sm.OLS(ts1, ts2).fit()

   c_t = ts.adfuller(result.resid)

   # Checking co-integration
   is_cointegrated = False
   threshold = 0.1
   ct4_ct0_difference = c_t[4][percent] - c_t[0]
   threshold_ct1_difference = threshold - c_t[1]
   if ct4_ct0_difference >= 0 and threshold_ct1_difference >= 0:
       is_cointegrated = True
       print( "Pair of securities are co-integrated ", "P_values: ", c_t[1], "Critical values: ", c_t[4], "T_values: ", c_t[0] )

   return is_cointegrated, result.params[0], result.params[1]





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
    smoothed_series = cointegrated_series.rolling(window=params["smoothing_window"]).mean()
    
    # Initialize signals list
    signals = []
    
    # Loop through the smoothed series
    for time, value in smoothed_series.items():
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




class Backtester:
    def __init__(self, price_data, signals, initial_capital=100000, transaction_cost=0.0):
        """
        Initialize the backtester with historical price data, trading signals, 
        initial capital, and transaction cost.
        
        Parameters:
        price_data (DataFrame): DataFrame containing historical closing prices for the pairs.
        signals (DataFrame): DataFrame containing the generated trading signals.
        initial_capital (float): Initial capital for the strategy (default is 100,000).
        transaction_cost (float): Transaction cost as a percentage of the trade value (default is 0.0).
        """
        self.price_data = price_data
        self.signals = signals
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Initialize variables to track the portfolio
        self.positions = pd.DataFrame(index=self.price_data.index, columns=self.price_data.columns)
        self.portfolio_value = pd.Series(index=self.price_data.index, name='Portfolio Value')
        self.cash = pd.Series(index=self.price_data.index, name='Cash')
        
        # Initialize the first day's cash and portfolio value
        self.cash.iloc[0] = self.initial_capital
        self.portfolio_value.iloc[0] = self.initial_capital
    
    def align_signals(self):
        """
        Align the trading signals with the price data and prepare for simulation.
        """
        # Merge the price data and signals to align them by date
        self.signals = self.signals.merge(self.price_data, on='Date', how='left')
    
    def simulate_trades(self):
        """
        Simulate the trades based on the generated signals.
        """
        for i in range(1, len(self.price_data)):
            current_date = self.price_data.index[i]
            previous_date = self.price_data.index[i-1]
            
            # Get the signals for the current date
            current_signals = self.signals.loc[current_date]
            
            # Iterate through each trading pair
            for pair in self.price_data.columns:
                signal = current_signals[pair]
                
                if signal == 'buy':
                    # If we don't already have a position, execute the buy
                    if self.positions.loc[previous_date, pair] == 0:
                        # Calculate the number of shares to buy
                        shares_to_buy = (self.cash.loc[previous_date] / (self.price_data.loc[current_date, pair] * (1 + self.transaction_cost)))
                        shares_to_buy = np.floor(shares_to_buy)  # Round down to the nearest whole share
                        
                        # Update positions
                        self.positions.loc[current_date, pair] = shares_to_buy
                        self.positions.loc[current_date, pair] += self.positions.loc[previous_date, pair]
                        
                        # Deduct cash for the trade
                        trade_cost = shares_to_buy * self.price_data.loc[current_date, pair] * (1 + self.transaction_cost)
                        self.cash.loc[current_date] = self.cash.loc[previous_date] - trade_cost
                        
                elif signal == 'sell':
                    # If we have a position, execute the sell
                    if self.positions.loc[previous_date, pair] > 0:
                        # Sell all shares
                        shares_to_sell = self.positions.loc[previous_date, pair]
                        
                        # Update positions
                        self.positions.loc[current_date, pair] = 0
                        
                        # Add cash from the sale
                        sale_revenue = shares_to_sell * self.price_data.loc[current_date, pair] * (1 - self.transaction_cost)
                        self.cash.loc[current_date] = self.cash.loc[previous_date] + sale_revenue
                        
            # Update portfolio value for the current date
            self.portfolio_value.loc[current_date] = self.cash.loc[current_date] + \
                                                        (self.positions.loc[current_date] * self.price_data.loc[current_date])


        return self.portfolio_value


    def calculate_performance_metrics(self):
            """
            Calculate performance metrics for the backtest.
            """
            self.returns = self.portfolio_value.pct_change()
            self.total_return = (self.returns + 1).cumprod().iloc[-1]
            self.max_drawdown = (self.portfolio_value.cummax().subtract(self.portfolio_value)).max() / self.portfolio_value.cummax().max()
            self.sharpe_ratio = self.returns.mean() / self.returns.std() * np.sqrt(252)
            self.annualized_volatility = self.returns.std() * np.sqrt(252)
            
            return {
                'Total Return': self.total_return,
                'Max Drawdown': self.max_drawdown,
                'Sharpe Ratio': self.sharpe_ratio,
                'Annualized Volatility': self.annualized_volatility
            }

    def visualize_equity_curve(self, save_results ):
        """
        Visualize the equity curve of the portfolio.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio_value)
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.savefig(f"{save_results}/equity_curve.png")

    def run_backtest(self, visualize  = True, save_results = "results"):
        """
        Run the complete backtest process.
        """
        self.align_signals()
        portfolio_value = self.simulate_trades()
        metrics = self.calculate_performance_metrics()
        if visualize:
            self.visualize_equity_curve( save_results )
        
        return portfolio_value, metrics


# # Fit the OLS model
# tested_pairs = set()
# cointegrated_tickers = {}

# for ticker_1 in crypto_tickers:
#     for ticker_2 in crypto_tickers:
#         if ticker_1 != ticker_2 and (ticker_2, ticker_1) not in tested_pairs:
#             result = sm.OLS(data[ticker_1], sm.add_constant(data[ticker_2]) ).fit()
#             c_t = ts.adfuller(result.resid)
#             if c_t[0]<= c_t[4]['10%'] and c_t[1]<= 0.1:

#                 print(f"Pair of securities {ticker_1} and {ticker_2} is co-integrated")
#                 print(f"p-value: {c_t[1]}")
#                 cointegrated_tickers.update({(ticker_1, ticker_2): result.params})
#             else:
#                 print(f"Pair of securities {ticker_1} and {ticker_2} is not co-integrated")
#             tested_pairs.add((ticker_1, ticker_2))






# for (pairs, params) in cointegrated_tickers.items():
#     # Create the cointegrated series
#     intercept = params[0]
#     beta = params[1]
#     cointegrated_series = data[pairs[0]] - (intercept + beta * data[pairs[1]])

#     # plot the cointegrated series
#     fig = plt.figure()
#     plt.plot(cointegrated_series)
#     plt.title(f"Cointegrated series of {pairs[0]} and {pairs[1]}")
#     plt.savefig(f"results/{pairs[0]}_{pairs[1]}_cointegrated_series.png")   








if __name__ == "__main__":

    data_fetcher = DataFetcher()

    # selected_symbols = ["ETHUSDT", "BTCUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT", "BCHUSDT"]
    selected_symbols = ["ETHUSDT", "BCHUSDT"]

    # Contract Type Contract Type
    # params = {'pair': symbol, 'contractType': contractType, 'interval': time_interval, 'limit': 96,
    #                   'startTime': start_time_since, 'endTime': loop_end_time}
    data_params={
        'pair': selected_symbols[0],
        'contractType': 'PERPETUAL',
        'interval': '1h',
        'limit': 96,
        'startTime': '2021-01-01',
        'endTime': '2023-12-02'
    }



    # Fetch the data
    pair_1 = data_fetcher.get_data(selected_symbols[0], params=data_params  )
    data_params["pair"] = selected_symbols[1]
    pair_2 = data_fetcher.get_data(selected_symbols[0], params=data_params  )

    # test for cointegration
    is_cointegrated, intercept, beta = adf_cointegration_test( pair_1['close'], pair_2['close'] )
    
    print("cointegration possibility: ", is_cointegrated)    

    cointegrated_series = pair_1['close'] - (intercept + beta * pair_2['close'])

    algorithm_params = {"lower_threshold":-2, "upper_threshold":2, "smoothing_window":10}
    
    # Generate signals
    signals = generate_trading_signals(cointegrated_series, params=algorithm_params )


    # Print the signals

    # Initialize backtester
    backtester = Backtester(
        price_data=price_data,
        signals=signals,
        initial_cash=100000,
        transaction_cost=0.002
    )

    # Add strategy
    backtester.add_strategy('example_strategy', signals )



