

# Import pandas
import pandas as pd
import yfinance as yf
import datetime
import time
import ccxt
from datetime import date
import numpy as np
# from datetime import time



import matplotlib.pyplot as plt



# import you













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



def find_cointegration_pairs( symbols: list[str]):
    data_params={
        'pair': selected_symbols[0],
        'contractType': 'PERPETUAL',
        'interval': '1h',
        'limit': 1000,
        'startTime': '2021-01-01',
        'endTime': '2024-12-02'
    }

    cointegrated_data = []
    for i, sym_1 in enumerate(symbols):
        for j, sym_2 in enumerate(symbols[i+1:]):
            # Fetch the data
            data_params["pair"] = sym_1
            pair_1_ = data_fetcher.get_data_as_array( request=data_params.copy()  )
            data_params["pair"] = sym_2
            pair_2_ = data_fetcher.get_data_as_array( request=data_params.copy()  )

            pair_1 = [ float(pair_1_[i][3]) for i in range(len(pair_1_)) ]
            pair_2 = [ float(pair_2_[i][3]) for i in range(len(pair_2_)) ]

            # test for cointegration
            is_cointegrated, intercept, beta = adf_cointegration_test( pair_1, pair_2 )

            if is_cointegrated:
                cointegrated_data.append( [pair_1, pair_2, intercept, beta, sym_1, sym_2] )       

    return cointegrated_data



if __name__ == "__main__":

    data_fetcher = DataFetcher()

    selected_symbols = ["ETHUSDT", "BTCUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT", "BCHUSDT"]
    # selected_symbols = ["ETHUSDT", "BCHUSDT"]

    # Contract Type Contract Type
    # params = {'pair': symbol, 'contractType': contractType, 'interval': time_interval, 'limit': 96,
    #                   'startTime': start_time_since, 'endTime': loop_end_time}
    data_params={
        'pair': selected_symbols[0],
        'contractType': 'PERPETUAL',
        'interval': '1h',
        'limit': 900,
        'startTime': '2021-01-01',
        'endTime': '2024-12-02'
    }

    paired_data = find_cointegration_pairs( selected_symbols )

    # test for cointegration
    # is_cointegrated, intercept, beta = adf_cointegration_test( paired_data[0][0], paired_data[0][1] )
    
    # print("cointegration possibility: ", is_cointegrated)    

    cointegrated_series = np.array(paired_data[0][0]) - (paired_data[0][2] + paired_data[0][3] * np.array(paired_data[0][1]) )

    plt.figure()
    plt.plot(cointegrated_series)
    plt.savefig(f"results/cointegrated_series_{paired_data[0][4]}_{paired_data[0][5]}.png")





    # test algorithm on the data
    data_params["pair"] = paired_data[0][4]
    pair_1_df = data_fetcher.get_data_as_dataframe( request=data_params.copy()  )
    data_params["pair"] = paired_data[0][5]
    pair_2_df = data_fetcher.get_data_as_dataframe( request=data_params.copy()  )

    price_data = pd.concat([pair_1_df['Close Price'], pair_2_df['Close Price']], axis=1)

    price_data['Cointegrated Series'] = pair_1_df['Close Price'] - (paired_data[0][2] + paired_data[0][3] * pair_2_df['Close Price'] )

    # make date index
    price_data['Date'] = pair_1_df['Open Time']
    price_data.set_index('Date', inplace=True)
    
    # Generate signals
    algorithm_params = {"lower_threshold":-2, "upper_threshold":2, "smoothing_window":10}
    signals = generate_trading_signals(price_data, params=algorithm_params )


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



