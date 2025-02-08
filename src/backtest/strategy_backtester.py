



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




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

