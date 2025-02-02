

# Import pandas
import pandas as pd
import yfinance as yf

import statsmodels.tsa.stattools as ts
import statsmodels.api as sm


import matplotlib.pyplot as plt




# Define list of tickers
crypto_tickers = [
    "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "DOGE-USD",
    "XRP-USD", "DOT-USD", "LTC-USD", "LINK-USD", "BCH-USD"
]


# Store the list in a Dataframe
data = pd.DataFrame(columns=crypto_tickers)

# Fetch the data
for ticker in crypto_tickers:
    data[ticker] = yf.download(ticker,'2022-07-02','2024-12-31', period="1h")['Close']

# Fit the OLS model
tested_pairs = set()
cointegrated_tickers = {}

for ticker_1 in crypto_tickers:
    for ticker_2 in crypto_tickers:
        if ticker_1 != ticker_2 and (ticker_2, ticker_1) not in tested_pairs:
            result = sm.OLS(data[ticker_1], sm.add_constant(data[ticker_2]) ).fit()
            c_t = ts.adfuller(result.resid)
            if c_t[0]<= c_t[4]['10%'] and c_t[1]<= 0.1:

                print(f"Pair of securities {ticker_1} and {ticker_2} is co-integrated")
                print(f"p-value: {c_t[1]}")
                cointegrated_tickers.update({(ticker_1, ticker_2): result.params})
            else:
                print(f"Pair of securities {ticker_1} and {ticker_2} is not co-integrated")
            tested_pairs.add((ticker_1, ticker_2))






for (pairs, params) in cointegrated_tickers.items():
    # Create the cointegrated series
    intercept = params[0]
    beta = params[1]
    cointegrated_series = data[pairs[0]] - (intercept + beta * data[pairs[1]])

    # plot the cointegrated series
    fig = plt.figure()
    plt.plot(cointegrated_series)
    plt.title(f"Cointegrated series of {pairs[0]} and {pairs[1]}")
    plt.savefig(f"results/{pairs[0]}_{pairs[1]}_cointegrated_series.png")   








