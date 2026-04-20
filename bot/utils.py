import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.dates import date2num
import matplotlib.colors as mcolors
import config

def load_and_clean_data(df):
    """
    Updated to handle unhashable list types before dropping duplicates.
    """
    df = df.copy()
    
    # 1. Convert any list columns (markers) to strings so they are hashable
    for col in ['ce_markers', 'pe_markers']:
        if col in df.columns:
            # Convert lists to string representation, e.g., "['ILLIQUID']"
            df[col] = df[col].astype(str)

    # 2. Now drop_duplicates will work correctly
    df = df.drop_duplicates()
    
    # 3. Standard cleaning
    df['fetch_time'] = pd.to_datetime(df['fetch_time']).dt.floor('30s')
    df['expiry'] = pd.to_datetime(df['expiry'])
    df['strike_price'] = df['strike_price'].astype(int)
    
    # Time to Expiry (in years)
    df['T'] = (df['expiry'] - df['fetch_time']).dt.total_seconds() / (365 * 24 * 3600)
    df = df[df['T'] > 0]

    # Instantaneous Liquidity
    df['liquidity'] = df[['ce_liveData_oi', 'pe_liveData_oi']].min(axis=1)

    # Liquidity filtering logic
    ce_marker = df['ce_markers'].str.upper()
    pe_marker = df['pe_markers'].str.upper()
    df['both_legs_liquid'] = (~ce_marker.str.contains('ILLIQUID')) & (~pe_marker.str.contains('ILLIQUID'))
    
    return df

def identify_arbitrage(df):
    """Identifies deviations from Put-Call Parity."""
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
    """Calculates total transaction friction for a single arbitrage leg."""
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

    # Funding and Margin costs
    funding = K * (np.exp(-config.rl * T) - np.exp(-config.rb * T))
    margin_block = (config.alpha + config.beta) * S
    margin_cost = margin_block * (np.exp(config.rb * T) - 1)

    # Total combined costs
    total_cost = (
        brk + txn + gst + stt +
        config.spread_cost +
        funding + margin_cost
    )

    return total_cost

def apply_costs(df):
    """Applies transaction costs to determine net profitability."""
    df['total_cost'] = df.apply(compute_cost, axis=1)
    df['net_profit_per_unit'] = df['abs_violation'] - df['total_cost']

    # profit per lot
    df['net_profit_per_lot'] = config.N * df['net_profit_per_unit']

    # Final trading signal
    df['is_profitable'] = df['net_profit_per_unit'] > 0

    return df

# Helper for visualization metrics
def _get_net_profit_series(data, trading_cost=True, per_lot=False):
    if trading_cost:
        net_profit = data['net_profit_per_unit'] if 'net_profit_per_unit' in data.columns else (np.abs(data['violation']) - data.apply(compute_cost, axis=1))
    else:
        net_profit = np.abs(data['violation'])

    return config.N * net_profit if per_lot else net_profit