import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    df['fetch_time'] = pd.to_datetime(df['fetch_time']).dt.floor('30s')
    df['expiry'] = pd.to_datetime(df['expiry'])
    df['strike_price'] = df['strike_price'].astype(int)
    df.drop_duplicates()
    
    # Time to Expiry (T in years)
    # T = (Expiry Date - Current Date) / 365
    df['T'] = (df['expiry'] - df['fetch_time']).dt.total_seconds() / (365 * 24 * 3600)
    df = df[df['T'] > 0]
    
    return df

def identify_arbitrage(df, r=0.07):
    # C - P
    df['market_side'] = df['ce_liveData_ltp'] - df['pe_liveData_ltp']
    
    # S - K * e^(-rT)
    df['theoretical_side'] = df['spot_price'] - (df['strike_price'] * np.exp(-r * df['T']))

    df['spread'] = df['market_side'] - df['theoretical_side']
    return df

def plot_arbitrage(df, strike):
    strike_df = df[df['strike_price'] == strike]
    plt.figure(figsize=(12, 6))
    plt.plot(strike_df['fetch_time'], strike_df['spread'], label='Arbitrage Spread')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Put-Call Parity Deviation for Strike {strike}')
    plt.ylabel('Spread (Points)')
    plt.legend()
    plt.show()