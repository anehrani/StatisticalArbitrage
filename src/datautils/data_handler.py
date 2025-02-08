


import ccxt
import pandas as pd
import time
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm




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

    def get_data_as_array( self, request:dict):
        # request should be like 
        # params = {'pair': symbol, 'contractType': contractType, 'interval': time_interval, 'limit': 96,
        #                   'startTime': start_time_since, 'endTime': loop_end_time}

        """
        # Column names
        columns = [
            'Open Time', 'Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume',
            'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
            'Taker Buy Quote Asset Volume', 'Ignore'
        ]
        """


        if 'startTime' in request:
            start_time = time.strptime(request['startTime'], '%Y-%m-%d')
            request['startTime'] = int(time.mktime(start_time) * 1000)
        if 'endTime' in request:
            end_time = time.strptime(request['endTime'], '%Y-%m-%d')
            request['endTime'] = int(time.mktime(end_time) * 1000)

        df = self.exchange.fapiPublicGetContinuousKlines(params=request)
        return df
    
    def get_data_as_dataframe(self, request: dict):
        # Fetch data from Binance API
        if 'startTime' in request:
            start_time = time.strptime(request['startTime'], '%Y-%m-%d')
            request['startTime'] = int(time.mktime(start_time) * 1000)
        if 'endTime' in request:
            end_time = time.strptime(request['endTime'], '%Y-%m-%d')
            request['endTime'] = int(time.mktime(end_time) * 1000)
        data = self.exchange.fapiPublicGetContinuousKlines(params=request)

        # Define column names for the DataFrame
        columns = [
            'Open Time', 'Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume',
            'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
            'Taker Buy Quote Asset Volume', 'Ignore'
        ]

        # Convert the data into a Pandas DataFrame
        df = pd.DataFrame(data, columns=columns)

        # Convert timestamp columns to human-readable dates
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

        # Convert numeric columns to appropriate data types
        numeric_columns = [
            'Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume',
            'Quote Asset Volume', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume'
        ]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

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
   ts2 = sm.add_constant(ts2)
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
   return is_cointegrated, result.params[0], result.params[1] 



