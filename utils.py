import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.dates import date2num
import config

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

def identify_arbitrage(df):

    # Market vs theoretical
    df['market_side'] = df['ce_liveData_ltp'] - df['pe_liveData_ltp']
    df['theoretical_side'] = df['spot_price'] - (
        df['strike_price'] * np.exp(-config.rl * df['T'])
    )

    # Raw violation
    df['violation'] = df['market_side'] - df['theoretical_side']

    # Case identification
    df['is_conversion'] = df['violation'] > 0   # C - P high
    df['is_reversal']   = df['violation'] < 0   # C - P low

    # Absolute violation magnitude
    df['abs_violation'] = np.abs(df['violation'])

    return df

def compute_cost(row):

    C = row['ce_liveData_ltp']
    P = row['pe_liveData_ltp']
    S = row['spot_price']
    K = row['strike_price']
    T = row['T']

    # Brokerage 
    brk = (
        min(config.tr * C, config.B) +
        min(config.tr * P, config.B) +
        min(config.tr * S, config.B)
    )

    # Transaction charges
    txn = (
        config.theta_c * C +
        config.theta_p * P +
        config.theta_s * S
    )

    # GST
    gst = config.gst * (brk + txn)

    # STT
    if row['is_conversion']:
        stt = config.tau_c_sell * C + config.tau_s * S
    elif row['is_reversal']:
        stt = config.tau_p_sell * P + config.tau_s * S
    else:
        stt = 0

    # Funding cost
    funding = K * (
        np.exp(-config.rl * T) - np.exp(-config.rb * T)
    )

    # Margin cost
    margin_block = (config.alpha + config.beta) * S
    margin_cost = margin_block * (np.exp(config.rb * T) - 1)

    # Total
    total_cost = (
        brk + txn + gst + stt +
        config.spread_cost +
        funding + margin_cost
    )

    return total_cost

def apply_costs(df):

    df['total_cost'] = df.apply(compute_cost, axis=1)
    df['net_profit_per_unit'] = df['abs_violation'] - df['total_cost']

    # profit per lot
    df['net_profit_per_lot'] = config.N * df['net_profit_per_unit']

    # Final signal
    df['is_profitable'] = df['net_profit_per_unit'] > 0

    return df

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