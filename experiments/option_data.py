
import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

def visualize_options_data(df):
    # Compute net long positions for each expiry: (bid volume - ask volume)
    df['net_long'] = df['longs'] - df['shorts']
    
    # Create a figure with two subplots:
    # 1. A bar chart for price change range vs expiry.
    # 2. A scatter plot correlating net long positions with price change range.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # --- Plot 1: Bar chart for price change range ---
    bars = ax1.bar(df['expiry'], df['price_change_range'], color='skyblue', edgecolor='black')
    ax1.set_title('Price Change Range vs Expiry')
    ax1.set_xlabel('Expiry Date')
    ax1.set_ylabel('Price Change Range (%)')
    
    # Annotate each bar with longs and shorts counts
    for idx, bar in enumerate(bars):
        longs = df.iloc[idx]['longs']
        shorts = df.iloc[idx]['shorts']
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                 f"L:{longs}\nS:{shorts}", ha='center', va='bottom', fontsize=8)
    
    # --- Plot 2: Scatter plot for Price Change Range vs Net Long ---
    ax2.scatter(df['net_long'], df['price_change_range'], 
                c='orange', edgecolors='black', s=100)
    ax2.set_title('Price Change Range vs Net Long (Bids - Asks)')
    ax2.set_xlabel('Net Long (Total Bid Volume - Total Ask Volume)')
    ax2.set_ylabel('Price Change Range (%)')
    
    # Optionally, annotate each point with its expiry date
    for i, row in df.iterrows():
        ax2.annotate(row['expiry'], (row['net_long'], row['price_change_range']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def get_option_data_from_deribit():
    """
    Fetches all BTC/USD options contracts from Deribit and returns a DataFrame with columns:
      - contract_date: The creation date of the contract (if available, otherwise current date)
      - expiry_date: The expiry date of the option derived from the expiration timestamp.
      - long: 1 if the option is a call (assumed bullish/long), otherwise 0.
      - short: 1 if the option is a put (assumed bearish/short), otherwise 0.
      - size: The contract size (if available).
      - volatility: Implied volatility (if available from the ticker data).
      - symbol: The trading symbol for the contract.
    """
    # Create exchange instance
    exchange = ccxt.deribit({
        'enableRateLimit': True,
    })
    
    print("Loading markets...")
    markets = exchange.load_markets()
    
    # Filter for option instruments from Deribit
    option_markets = [m for m in markets.values() if m.get('type') == 'option']
    print(f"Found {len(option_markets)} option instruments.")
    
    # Limit to BTC/USD instruments.
    btc_usd_options = []
    for market in option_markets:
        # Some markets may not explicitly list 'base' or 'quote'.
        if market.get('base') == 'BTC' and market.get('quote') == 'USD':
            btc_usd_options.append(market)
            
    print(f"Filtered to {len(btc_usd_options)} BTC/USD option instruments.")
    
    records = []
    for market in btc_usd_options:
        symbol = market['symbol']
        try:
            # Get contract (creation) date if available; if not, use current date.
            creation_ts = market['info'].get('creation_timestamp')
            if creation_ts:
                contract_date = datetime.utcfromtimestamp(creation_ts / 1000).strftime("%Y-%m-%d")
            else:
                contract_date = datetime.utcnow().strftime("%Y-%m-%d")
            
            # Get expiry date from expiration_timestamp (assumed to be in ms)
            exp_ts = market['info'].get('expiration_timestamp')
            if exp_ts:
                expiry_date = datetime.utcfromtimestamp(exp_ts / 1000).strftime("%Y-%m-%d")
            else:
                expiry_date = None

            # Determine option type from market info: assume option_type is provided:
            # if option_type == 'call' then long flag 1, if 'put' then short flag 1.
            option_type = market['info'].get('option_type', '').lower()
            long_flag = 1 if option_type == 'call' else 0
            short_flag = 1 if option_type == 'put' else 0
            
            # Size: try to get the contract size (can be at market level or within info)
            size = market.get('contractSize') or market['info'].get('contract_size')
            if size is None:
                size = 0  # if not available
            
            # Fetch ticker data in order to get implied volatility
            ticker = exchange.fetch_ticker(symbol)
            # 'iv' field might be available or alternatively 'impliedVolatility'
            volatility = ticker.get('iv') or ticker.get('impliedVolatility') or 0.0
            # You can adjust the scaling (Deribit's iv is usually expressed as a percentage)
            
            records.append({
                'contract_date': contract_date,
                'expiry_date': expiry_date,
                'long': long_flag,
                'short': short_flag,
                'size': size,
                'volatility': volatility,
                'symbol': symbol
            })
            print(f"Processed {symbol}: contract date {contract_date}, expiry {expiry_date}, "
                  f"type {option_type}, size {size}, volatility {volatility:.2f}")
            
            # Pause to avoid rate limits.
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    if not records:
        print("No BTC/USD option contracts found.")
        return None

    # Convert records to DataFrame
    df = pd.DataFrame(records)
    return df


import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
import time

def visualize_options_data_3d(df):
    """
    3D visualization for BTC/USD options contracts.
    X-axis: Last Price
    Y-axis: Expiry Date (as numeric date)
    Z-axis: Contract Type (0 for Call, 1 for Put)
    """
    # Convert expiry_date strings to datetime and then use matplotlib date numbers
    df['expiry_dt'] = pd.to_datetime(df['expiry_date'])
    df['expiry_num'] = df['expiry_dt'].apply(mdates.date2num)
    
    # Create a categorical mapping for contract type:
    # 0 -> Call, 1 -> Put
    def get_contract_type(row):
        if row['long'] == 1:
            return "Call"
        elif row['short'] == 1:
            return "Put"
        else:
            return "Unknown"
    
    df['contract_type'] = df.apply(get_contract_type, axis=1)
    # Map contract type to numeric values for the Z-axis:
    type_mapping = {"Call": 0, "Put": 1, "Unknown": -1}
    df['contract_numeric'] = df['contract_type'].map(type_mapping)
    
    # Create the 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df['price'], 
        df['expiry_num'], 
        df['contract_numeric'],
        c=df['contract_numeric'], cmap='viridis', s=100, edgecolor='black'
    )
    
    ax.set_xlabel('Last Price')
    ax.set_ylabel('Expiry Date')
    ax.set_zlabel('Contract Type (0=Call, 1=Put)')
    ax.set_title('3D Visualization of BTC/USD Option Contracts')
    
    # Format the y-axis to display dates
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    date_nums = df['expiry_num'].tolist()
    ax.set_yticks(date_nums)
    
    # Create a legend manually for contract type
    import matplotlib.patches as mpatches
    call_patch = mpatches.Patch(color=plt.cm.viridis(0.0), label="Call")
    put_patch = mpatches.Patch(color=plt.cm.viridis(1.0), label="Put")
    ax.legend(handles=[call_patch, put_patch])
    
    plt.show()

def get_option_data_from_deribit():
    """
    Fetches all BTC/USD options contracts from Deribit and returns a DataFrame with columns:
      - contract_date: The creation date of the contract (if available, otherwise current date)
      - expiry_date: The expiry date of the option derived from the expiration timestamp.
      - long: 1 if the option is a call, otherwise 0.
      - short: 1 if the option is a put, otherwise 0.
      - size: The contract size (if available).
      - volatility: Implied volatility (if available from the ticker data).
      - price: Last traded price from the ticker.
      - symbol: The trading symbol for the contract.
    """
    exchange = ccxt.deribit({
        'enableRateLimit': True,
    })

    print("Loading markets...")
    markets = exchange.load_markets()

    option_markets = [m for m in markets.values() if m.get('type') == 'option']
    print(f"Found {len(option_markets)} option instruments.")

    # Limit to BTC/USD instruments.
    btc_usd_options = []
    for market in option_markets:
        if market.get('base') == 'BTC' and market.get('quote') == 'USD':
            btc_usd_options.append(market)
    print(f"Filtered to {len(btc_usd_options)} BTC/USD option instruments.")

    records = []
    for market in btc_usd_options:
        symbol = market['symbol']
        try:
            # Contract (creation) date or use current date if not available.
            creation_ts = market['info'].get('creation_timestamp')
            if creation_ts:
                contract_date = datetime.utcfromtimestamp(creation_ts / 1000).strftime("%Y-%m-%d")
            else:
                contract_date = datetime.utcnow().strftime("%Y-%m-%d")
            
            # Expiry date from expiration_timestamp (ms).
            exp_ts = market['info'].get('expiration_timestamp')
            expiry_date = datetime.utcfromtimestamp(exp_ts / 1000).strftime("%Y-%m-%d") if exp_ts else None

            # Option type: assume field `option_type` exists.
            option_type = market['info'].get('option_type', '').lower()
            long_flag = 1 if option_type == 'call' else 0
            short_flag = 1 if option_type == 'put' else 0
            
            # Contract size
            size = market.get('contractSize') or market['info'].get('contract_size')
            if size is None:
                size = 0
            
            # Fetch ticker for volatility and price data.
            ticker = exchange.fetch_ticker(symbol)
            volatility = ticker.get('iv') or ticker.get('impliedVolatility') or 0.0
            price = ticker.get('last') or 0.0
            
            records.append({
                'contract_date': contract_date,
                'expiry_date': expiry_date,
                'long': long_flag,
                'short': short_flag,
                'size': size,
                'volatility': volatility,
                'price': price,
                'symbol': symbol
            })
            print(f"Processed {symbol}: contract date {contract_date}, expiry {expiry_date}, "
                  f"type {option_type}, size {size}, price {price}, volatility {volatility:.2f}")
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    if not records:
        print("No BTC/USD option contracts found.")
        return None

    df = pd.DataFrame(records)
    return df

def trade_signal(df):
    """
    Implements a basic trading algorithm based on aggregated option contract data.
    
    Trading Logic:
      1. Calculate total contract size for calls and puts.
      2. Compute a call_ratio: calls_total / (calls_total + puts_total).
      3. Check if the average implied volatility is in a "moderate" range.
         - If call_ratio > 0.60 and volatility is moderate => Signal BUY (bullish bias).
         - If call_ratio < 0.40 and volatility is moderate => Signal SELL (bearish bias).
         - Otherwise, signal HOLD.
    
    Returns:
      A dictionary containing the aggregated data and the trade signal.
    """
    # Sum contract sizes for calls and puts
    calls_total = df.loc[df['long'] == 1, 'size'].sum()
    puts_total = df.loc[df['short'] == 1, 'size'].sum()
    total_size = calls_total + puts_total

    if total_size == 0:
        return {"trade_decision": "HOLD", "reason": "No aggregated contract size."}

    call_ratio = calls_total / total_size

    # Compute average volatility (in percentage if expressed as such)
    avg_volatility = df['volatility'].mean()

    # These thresholds are for demonstration purposes.
    # You can adjust the thresholds based on empirical analysis or risk tolerance.
    if call_ratio > 0.60 and 20 <= avg_volatility <= 80:
        decision = "BUY"
        reason = f"Call ratio high ({call_ratio:.2f}) and average volatility moderate ({avg_volatility:.2f}%)."
    elif call_ratio < 0.40 and 20 <= avg_volatility <= 80:
        decision = "SELL"
        reason = f"Call ratio low ({call_ratio:.2f}) and average volatility moderate ({avg_volatility:.2f}%)."
    else:
        decision = "HOLD"
        reason = f"Uncertain market: call ratio ({call_ratio:.2f}), average volatility ({avg_volatility:.2f}%)."

    result = {
        "trade_decision": decision,
        "calls_total": calls_total,
        "puts_total": puts_total,
        "call_ratio": call_ratio,
        "avg_volatility": avg_volatility,
        "reason": reason
    }
    return result

def main():
    df_options = get_option_data_from_deribit()
    if df_options is not None and not df_options.empty:
        print("\nFetched options contract data:")
        print(df_options)
        visualize_options_data_3d(df_options)
        
        signal = trade_signal(df_options)
        print("\nTrade Signal:")
        print(signal)
        
        # Depending on the live trading integration, here you could place orders, send alerts, etc.
    else:
        print("No data to display.")

if __name__ == '__main__':
    main()

