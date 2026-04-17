import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.dates import date2num

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    df['fetch_time'] = pd.to_datetime(df['fetch_time']).dt.floor('30s')
    df['expiry'] = pd.to_datetime(df['expiry'])
    df['strike_price'] = df['strike_price'].astype(int)
    df = df.drop_duplicates()
    
    # Time to Expiry (in years)
    df['T'] = (df['expiry'] - df['fetch_time']).dt.total_seconds() / (365 * 24 * 3600)
    df = df[df['T'] > 0]

    return df

import numpy as np
import pandas as pd

def identify_arbitrage(df, r=0.07, avoid_illiquid=False, oi_threshold=500):
    df['market_side'] = df['ce_liveData_ltp'] - df['pe_liveData_ltp']
    df['theoretical_side'] = df['spot_price'] - (df['strike_price'] * np.exp(-r * df['T']))
    df['spread'] = df['market_side'] - df['theoretical_side']
    
    # 2. Market Implied Interest Rate (r_market)
    S = df['spot_price']
    C = df['ce_liveData_ltp']
    P = df['pe_liveData_ltp']
    K = df['strike_price']
    T = df['T']

    # num = S - (C - P)
    num = S - df['market_side']
    valid_log = (num > 0) & (K > 0) & (T > 0)

    df['r_market'] = np.nan
    df.loc[valid_log, 'r_market'] = - (1 / T[valid_log]) * np.log(num[valid_log] / K[valid_log])

    is_marked_illiquid = df['ce_markers'].astype(str).str.contains('ILLIQUID', na=False)
    below_oi_cutoff = df['ce_liveData_oi'] < oi_threshold

    if avoid_illiquid:
        invalid_mask = is_marked_illiquid | below_oi_cutoff
    else:
        invalid_mask = below_oi_cutoff

    # Apply the mask to relevant columns
    df.loc[invalid_mask, ['spread', 'r_market']] = np.nan

    return df

def compute_cost(row):
    premium = row['ce_liveData_ltp'] + row['pe_liveData_ltp']   # approx turnover

    brokerage = 3 * 20

    stt = 0.0015 * premium   # sell side approx
    txn = 0.0003553 * premium
    sebi = 0.000001 * premium
    stamp = 0.00003 * premium  # only buy side

    gst = 0.18 * (brokerage + txn + sebi)

    return brokerage + stt + txn + sebi + stamp + gst

def plot_arbitrage(df, strike, avoid_illiquid=False, oi_threshold=500, trading_cost = True):
    data = df[df['strike_price'] == strike].sort_values('fetch_time').copy()

    x = data['fetch_time'].values

    if trading_cost:
        cost = data.apply(compute_cost, axis=1)
        y = data['spread'] - cost
    else:
        y = data['spread']

    points = np.array([date2num(x), y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = []
    for i in range(len(data) - 1):
        row = data.iloc[i]
        is_marked_illiquid = 'ILLIQUID' in str(row['ce_markers'])
        oi_value = row['ce_liveData_oi']
        
        # Priority 1: Explicit 'ILLIQUID' Marker (Red)
        if avoid_illiquid and is_marked_illiquid:
            colors.append('red')
        # Priority 2: Below Liquidity Threshold (Yellow)
        elif oi_value < oi_threshold:
            colors.append('yellow')
        # Priority 3: Liquid (Green)
        else:
            colors.append('green')

    fig, ax = plt.subplots(figsize=(14, 7))
    lc = LineCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)
    
    # Auto-scale the axes
    ax.autoscale_view()
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Formatting
    plt.title(f"Arbitrage Spread for Strike {strike} (Threshold: {oi_threshold} OI)")
    plt.ylabel("Spread (Market - Theoretical)")
    plt.xlabel("Time")
    
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Liquid'),
        Line2D([0], [0], color='yellow', lw=2, label='Low Liquidity'),
        Line2D([0], [0], color='red', lw=2, label='Illiquid Marker')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()