import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.dates import date2num
import matplotlib.colors as mcolors
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

    # --- NEW ADDITION ---
    # Instantaneous Liquidity (Bottleneck Open Interest)
    # Takes the minimum of Call OI and Put OI for every row instantly
    df['liquidity'] = df[['ce_liveData_oi', 'pe_liveData_oi']].min(axis=1)

    # Per-row liquidity state and per-strike % time both legs are liquid.
    ce_marker = df['ce_markers'].fillna('').astype(str).str.upper()
    pe_marker = df['pe_markers'].fillna('').astype(str).str.upper()
    df['both_legs_liquid'] = (~ce_marker.str.contains('ILLIQUID')) & (~pe_marker.str.contains('ILLIQUID'))
    df['both_legs_liquid_time_pct'] = (
        df.groupby('strike_price')['both_legs_liquid'].transform('mean') * 100.0
    )

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


def compute_cost_breakdown(row):

    C = row['ce_liveData_ltp']
    P = row['pe_liveData_ltp']
    S = row['spot_price']
    K = row['strike_price']
    T = row['T']

    brokerage = (
        min(config.tr * C, config.B) +
        min(config.tr * P, config.B) +
        min(config.tr * S, config.B)
    )

    transaction_charges = (
        config.theta_c * C +
        config.theta_p * P +
        config.theta_s * S
    )

    gst = config.gst * (brokerage + transaction_charges)

    if row['is_conversion']:
        stt = config.tau_c_sell * C + config.tau_s * S
    elif row['is_reversal']:
        stt = config.tau_p_sell * P + config.tau_s * S
    else:
        stt = 0

    funding = K * (np.exp(-config.rl * T) - np.exp(-config.rb * T))

    margin_block = (config.alpha + config.beta) * S
    margin_cost = margin_block * (np.exp(config.rb * T) - 1)

    total_cost = (
        brokerage + transaction_charges + gst + stt +
        config.spread_cost + funding + margin_cost
    )

    return pd.Series({
        'brokerage': brokerage,
        'transaction_charges': transaction_charges,
        'gst': gst,
        'stt': stt,
        'funding': funding,
        'margin_cost': margin_cost,
        'spread_cost': config.spread_cost,
        'total_cost': total_cost,
    })


def plot_avg_cost_breakdown_pie(df, profitable_only=True, min_liquidity=0, per_lot=False):
    data = df.copy()

    if min_liquidity > 0:
        data = data[data['liquidity'] >= min_liquidity].copy()

    if 'net_profit_per_unit' not in data.columns:
        data['net_profit_per_unit'] = _get_net_profit_series(data, trading_cost=True, per_lot=False)

    if profitable_only:
        data = data[data['net_profit_per_unit'] > 0].copy()

    if data.empty:
        raise ValueError('No rows available for the selected filter')

    breakdown = data.apply(compute_cost_breakdown, axis=1)
    avg = breakdown.mean()

    scale = config.N if per_lot else 1.0

    # Keep the requested 4-way split while preserving full total cost accounting.
    slices = pd.Series({
        'Brokerage': (avg['brokerage'] + avg['transaction_charges'] + avg['spread_cost']) * scale,
        'STT': avg['stt'] * scale,
        'GST': avg['gst'] * scale,
        'Funding/Margin': (avg['funding'] + avg['margin_cost']) * scale,
    })

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#59a14f']
    explode = [0.02, 0.02, 0.02, 0.02]
    ax.pie(
        slices.values,
        labels=slices.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        pctdistance=0.75,
    )
    ax.axis('equal')

    unit_label = 'Lot' if per_lot else 'Unit'
    title_filter = 'Profitable Windows' if profitable_only else 'All Windows'
    ax.set_title(f'Average Total Cost Breakdown ({title_filter})\nper {unit_label}')

    plt.tight_layout()
    plt.show()

    out = slices.to_frame(name=f'avg_cost_per_{unit_label.lower()}')
    out.loc['Total', f'avg_cost_per_{unit_label.lower()}'] = slices.sum()
    return out

def apply_costs(df):

    df['total_cost'] = df.apply(compute_cost, axis=1)
    df['net_profit_per_unit'] = df['abs_violation'] - df['total_cost']

    # profit per lot
    df['net_profit_per_lot'] = config.N * df['net_profit_per_unit']

    # Final signal
    df['is_profitable'] = df['net_profit_per_unit'] > 0

    return df


def _get_abs_violation_series(data):
    if 'abs_violation' in data.columns:
        return data['abs_violation']
    if 'violation' in data.columns:
        return data['violation'].abs()
    if 'spread' in data.columns:
        return data['spread'].abs()
    raise KeyError("Expected one of ['abs_violation', 'violation', 'spread']")


def _get_liquidity_series(data):
    if 'liquidity' in data.columns:
        return data['liquidity']
    return data[['ce_liveData_oi', 'pe_liveData_oi']].min(axis=1)


def _apply_intraday_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def _set_strike_ticks(ax, strike_series, step=500):
    min_strike = int(strike_series.min())
    max_strike = int(strike_series.max())
    start_tick = (min_strike // step) * step
    end_tick = ((max_strike + step - 1) // step) * step
    ax.set_xticks(np.arange(start_tick, end_tick + 1, step))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def _get_net_profit_series(data, trading_cost=True, per_lot=False):
    abs_violation = _get_abs_violation_series(data)

    if trading_cost:
        if 'net_profit_per_unit' in data.columns:
            net_profit = data['net_profit_per_unit']
        else:
            net_profit = abs_violation - data.apply(compute_cost, axis=1)
    else:
        net_profit = abs_violation

    if per_lot:
        return config.N * net_profit

    return net_profit

def plot_arbitrage(df, strike, avoid_illiquid=False, oi_threshold=500, trading_cost=True):
    data = df[df['strike_price'] == strike].sort_values('fetch_time').copy()

    if data.empty:
        print(f"No data found for strike {strike}")
        return

    x = data['fetch_time'].values
    y = _get_net_profit_series(data, trading_cost=trading_cost, per_lot=False)

    # Convert timestamps to matplotlib-compatible numbers
    x_nums = date2num(x)
    points = np.array([x_nums, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = []
    for i in range(len(data) - 1):
        row = data.iloc[i]
        is_marked_illiquid = 'ILLIQUID' in str(row['ce_markers']) or 'ILLIQUID' in str(row['pe_markers'])
        oi_value = row['liquidity'] 
        
        if avoid_illiquid and is_marked_illiquid:
            colors.append('red')
        elif oi_value < oi_threshold:
            colors.append('yellow')
        else:
            colors.append('green')

    fig, ax = plt.subplots(figsize=(14, 7))
    lc = LineCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)
    
    _apply_intraday_time_axis(ax)

    # Auto-scale and baseline
    ax.autoscale_view()
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Formatting
    plt.title(f"Arbitrage Net Profit: Strike {strike} (Threshold: {oi_threshold} OI)")
    plt.ylabel('Net Profit (Post-Cost)' if trading_cost else 'Gross Violation Magnitude')
    plt.xlabel("Time of Day (April 6th)")
    
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Liquid (Both Legs)'),
        Line2D([0], [0], color='yellow', lw=2, label='Low Liquidity Bottleneck'),
        Line2D([0], [0], color='red', lw=2, label='Illiquid Marker')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, which='both', linestyle=':', alpha=0.3) # Added a light grid for readability
    plt.tight_layout()
    plt.show()

def plot_liquidity(df, strike, oi_threshold=500):
    # 1. Filter for your specific strike price
    data = df[df['strike_price'] == strike].copy()

    if data.empty:
        raise ValueError(f"No rows found for strike {strike}")
    
    # 2. Sort by time just in case the data is out of order
    data = data.sort_values('fetch_time')

    # 3. Plot Time vs Liquidity (bottleneck OI)
    liquidity = _get_liquidity_series(data)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data['fetch_time'], liquidity, color='blue', linewidth=2)

    _apply_intraday_time_axis(ax)
    
    # Formatting
    plt.title(f"Instantaneous Liquidity (Bottleneck OI) over Time - Strike {strike}")
    plt.xlabel("Time")
    plt.ylabel("Maximum Tradable Contracts (Min OI)")
    
    # 4. Use the parameter for the threshold line
    plt.axhline(y=oi_threshold, color='red', linestyle='--', label=f"Liquidity Threshold ({oi_threshold})") 
    
    plt.grid(True, which='both', linestyle=':', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_arbitrage_and_liquidity(df, strike, oi_threshold=500, trading_cost=True):
    data = df[df['strike_price'] == strike].sort_values('fetch_time').copy()

    if data.empty:
        raise ValueError(f"No rows found for strike {strike}")

    x = data['fetch_time'].values
    arbitrage_y = _get_net_profit_series(data, trading_cost=trading_cost, per_lot=False)

    liquidity_y = _get_liquidity_series(data)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    line1, = ax1.plot(x, arbitrage_y, color='tab:blue', linewidth=2, label='Arbitrage')
    line2, = ax2.plot(x, liquidity_y, color='tab:orange', linewidth=2, label='Liquidity (Bottleneck OI)')

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    threshold_line = ax2.axhline(
        y=oi_threshold,
        color='tab:red',
        linestyle='--',
        alpha=0.8,
        label=f'Liquidity Threshold ({oi_threshold})'
    )

    _apply_intraday_time_axis(ax1)

    ax1.set_title(f'Arbitrage and Liquidity Over Time - Strike {strike}')
    ax1.set_xlabel('Time of Day (April 6th)')
    ax1.set_ylabel('Net Profit (Post-Cost)' if trading_cost else 'Gross Violation Magnitude')
    ax2.set_ylabel('Liquidity (Min of CE/PE OI)')

    ax1.legend([line1, line2, threshold_line], [
        line1.get_label(),
        line2.get_label(),
        threshold_line.get_label(),
    ], loc='upper right')

    ax1.grid(True, which='both', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spot_price(df):
    data = df.sort_values('fetch_time').copy()

    if data.empty:
        raise ValueError("No rows available to plot spot price")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data['fetch_time'], data['spot_price'], color='tab:green', linewidth=2, label='Spot Price')

    _apply_intraday_time_axis(ax)

    ax.set_title('Spot Price Over Time')
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Spot Price')
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_spot_and_max_profit_over_time(
    df,
    min_liquidity=0,
    trading_cost=True,
    per_lot=False,
    figsize=(14, 8),
):
    data = df.copy()

    if min_liquidity > 0:
        data = data[data['liquidity'] >= min_liquidity].copy()

    if data.empty:
        raise ValueError('No rows available after filtering')

    data['profit_value'] = _get_net_profit_series(
        data,
        trading_cost=trading_cost,
        per_lot=per_lot,
    )

    spot_series = (
        data.groupby('fetch_time', as_index=False)['spot_price']
        .mean()
        .sort_values('fetch_time')
        .rename(columns={'spot_price': 'spot_price'})
    )

    max_profit_series = (
        data.groupby('fetch_time', as_index=False)['profit_value']
        .max()
        .sort_values('fetch_time')
        .rename(columns={'profit_value': 'max_profit'})
    )

    combined = spot_series.merge(max_profit_series, on='fetch_time', how='inner')
    if combined.empty:
        raise ValueError('No overlapping time points found for plotting')

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1]},
    )

    ax_top.plot(
        combined['fetch_time'],
        combined['spot_price'],
        color='tab:green',
        linewidth=2,
        label='Nifty Spot Price',
    )
    ax_top.set_title('Nifty Spot Price and Max Net Profit Over Time')
    ax_top.set_ylabel('Spot Price')
    ax_top.grid(True, which='both', linestyle=':', alpha=0.3)
    ax_top.legend(loc='upper right')

    ax_bottom.plot(
        combined['fetch_time'],
        combined['max_profit'],
        color='tab:blue',
        linewidth=2,
        label='Max Net Profit Across Strikes',
    )
    ax_bottom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax_bottom.set_ylabel('Max Net Profit per Lot (₹)' if per_lot else 'Max Net Profit per Unit (₹)')
    ax_bottom.set_xlabel('Time of Day')
    ax_bottom.grid(True, which='both', linestyle=':', alpha=0.3)
    ax_bottom.legend(loc='upper right')

    _apply_intraday_time_axis(ax_bottom)

    plt.tight_layout()
    plt.show()

    return combined


def plot_cumsum_arbitrage_profit_all_opportunities(
    df,
    require_liquid=True,
    min_liquidity=0,
    per_lot=True,
    figsize=(14, 6),
):
    """
    Plot cumulative potential arbitrage profit for the full day.

    Assumption:
    - At every profitable arbitrage opportunity, exactly 1 lot is traded.
    - Opportunities are taken across all strikes at each timestamp.
    """
    data = df.copy()

    if min_liquidity > 0:
        data = data[data['liquidity'] >= min_liquidity].copy()

    if require_liquid and 'both_legs_liquid' in data.columns:
        data = data[data['both_legs_liquid']].copy()

    if data.empty:
        raise ValueError('No rows available after applying filters')

    data['profit_value'] = _get_net_profit_series(
        data,
        trading_cost=True,
        per_lot=per_lot,
    )

    opportunities = data[data['profit_value'] > 0].copy()
    if opportunities.empty:
        raise ValueError('No profitable arbitrage opportunities found')

    summary = (
        opportunities.groupby('fetch_time', as_index=False)
        .agg(
            interval_profit=('profit_value', 'sum'),
            opportunities_count=('profit_value', 'size'),
        )
        .sort_values('fetch_time')
    )

    summary['cumulative_profit'] = summary['interval_profit'].cumsum()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        summary['fetch_time'],
        summary['cumulative_profit'],
        color='tab:green',
        linewidth=2,
        label='Cumulative Arbitrage Profit',
    )

    _apply_intraday_time_axis(ax)
    ax.set_title('Cumulative Potential Arbitrage Profit (All Profitable Opportunities)')
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Cumulative Profit (₹ per Lot)' if per_lot else 'Cumulative Profit (₹ per Unit)')
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    ax.legend(loc='upper left')

    total_profit = summary['cumulative_profit'].iloc[-1]
    ax.text(
        0.01,
        0.95,
        f'Total Day Profit: {total_profit:,.2f}',
        transform=ax.transAxes,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    return summary


def plot_strike_vs_both_legs_liquid_pct(df):
    required_cols = {'strike_price', 'both_legs_liquid_time_pct'}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    summary = (
        df.groupby('strike_price', as_index=False)['both_legs_liquid_time_pct']
        .first()
        .sort_values('strike_price')
    )
    summary = summary.rename(columns={'both_legs_liquid_time_pct': 'both_liquid_pct'})

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(summary['strike_price'], summary['both_liquid_pct'], color='tab:purple', linewidth=2, marker='o')

    ax.set_title('Strike Price vs % Time Both CE and PE Are Liquid')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Both-Legs Liquid Time (%)')
    ax.set_ylim(0, 100)

    _set_strike_ticks(ax, summary['strike_price'])

    ax.grid(True, which='both', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return summary[['strike_price', 'both_liquid_pct']]


def plot_strike_vs_max_arbitrage(df, trading_cost=True, per_lot=False):
    data = df.copy()
    data['profit_value'] = _get_net_profit_series(
        data,
        trading_cost=trading_cost,
        per_lot=per_lot,
    )

    summary = (
        data.groupby('strike_price', as_index=False)['profit_value']
        .max()
        .sort_values('strike_price')
    )
    summary = summary.rename(columns={'profit_value': 'max_profit'})

    max_abs = summary['max_profit'].abs().max()
    if max_abs > 0:
        summary['max_profit_norm_pct'] = (summary['max_profit'] / max_abs) * 100.0
    else:
        summary['max_profit_norm_pct'] = 0.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(summary['strike_price'], summary['max_profit_norm_pct'], color='tab:blue', linewidth=2, marker='o')

    ax.set_title('Strike Price vs Max Net Profit (Normalized 0-100)')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Max Net Profit Normalized (%)')
    ax.set_ylim(0, 100)

    _set_strike_ticks(ax, summary['strike_price'])

    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return summary[['strike_price', 'max_profit', 'max_profit_norm_pct']]


def plot_strike_vs_max_arbitrage_and_liquidity_pct(df, trading_cost=True, per_lot=False):
    required_cols = {'strike_price', 'both_legs_liquid_time_pct'}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data['profit_value'] = _get_net_profit_series(
        data,
        trading_cost=trading_cost,
        per_lot=per_lot,
    )

    arb_summary = (
        data.groupby('strike_price', as_index=False)['profit_value']
        .max()
        .rename(columns={'profit_value': 'max_profit'})
    )
    liq_summary = (
        data.groupby('strike_price', as_index=False)['both_legs_liquid_time_pct']
        .first()
        .rename(columns={'both_legs_liquid_time_pct': 'both_liquid_pct'})
    )

    summary = (
        arb_summary.merge(liq_summary, on='strike_price', how='inner')
        .sort_values('strike_price')
    )

    max_abs = summary['max_profit'].abs().max()
    if max_abs > 0:
        summary['max_profit_norm_pct'] = (summary['max_profit'] / max_abs) * 100.0
    else:
        summary['max_profit_norm_pct'] = 0.0

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    line1, = ax1.plot(
        summary['strike_price'],
        summary['max_profit_norm_pct'],
        color='tab:blue',
        linewidth=2,
        marker='o',
        label='Max Net Profit (Normalized %)',
    )
    line2, = ax2.plot(
        summary['strike_price'],
        summary['both_liquid_pct'],
        color='tab:purple',
        linewidth=2,
        marker='o',
        label='% Time Both CE/PE Liquid',
    )

    ax1.set_title('Strike Price vs Max Net Profit (Normalized) and % Time Both CE/PE Liquid')
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Max Net Profit Normalized (%)')
    ax1.set_ylim(0, 100)
    ax2.set_ylabel('Both-Legs Liquid Time (%)')
    ax2.set_ylim(0, 100)

    _set_strike_ticks(ax1, summary['strike_price'])

    ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc='upper right')
    ax1.grid(True, which='both', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return summary[['strike_price', 'max_profit', 'max_profit_norm_pct', 'both_liquid_pct']]


def plot_strike_vs_arbitrage_positive_time_pct(df, trading_cost=True, per_lot=False):
    data = df.copy()
    data['profit_value'] = _get_net_profit_series(
        data,
        trading_cost=trading_cost,
        per_lot=per_lot,
    )

    data['arbitrage_positive'] = data['profit_value'] > 0

    summary = (
        data.groupby('strike_price', as_index=False)['arbitrage_positive']
        .mean()
        .sort_values('strike_price')
    )
    summary['arbitrage_positive_time_pct'] = summary['arbitrage_positive'] * 100.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        summary['strike_price'],
        summary['arbitrage_positive_time_pct'],
        color='tab:blue',
        linewidth=2,
        marker='o',
    )

    ax.set_title('Strike Price vs % Time Net Profit > 0')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Net Profit > 0 Time (%)')
    ax.set_ylim(0, 100)

    _set_strike_ticks(ax, summary['strike_price'])

    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return summary[['strike_price', 'arbitrage_positive_time_pct']]


def plot_strike_vs_arbitrage_positive_and_liquidity_pct(df, trading_cost=True, per_lot=False):
    required_cols = {'strike_price', 'both_legs_liquid_time_pct'}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data['profit_value'] = _get_net_profit_series(
        data,
        trading_cost=trading_cost,
        per_lot=per_lot,
    )

    data['arbitrage_positive'] = data['profit_value'] > 0

    arb_summary = (
        data.groupby('strike_price', as_index=False)['arbitrage_positive']
        .mean()
        .rename(columns={'arbitrage_positive': 'arbitrage_positive_pct'})
    )
    arb_summary['arbitrage_positive_pct'] = arb_summary['arbitrage_positive_pct'] * 100.0

    liq_summary = (
        data.groupby('strike_price', as_index=False)['both_legs_liquid_time_pct']
        .first()
        .rename(columns={'both_legs_liquid_time_pct': 'both_liquid_pct'})
    )

    summary = (
        arb_summary.merge(liq_summary, on='strike_price', how='inner')
        .sort_values('strike_price')
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        summary['strike_price'],
        summary['arbitrage_positive_pct'],
        color='tab:blue',
        linewidth=2,
        marker='o',
        label='% Time Net Profit > 0',
    )
    ax.plot(
        summary['strike_price'],
        summary['both_liquid_pct'],
        color='tab:purple',
        linewidth=2,
        marker='o',
        label='% Time Both CE/PE Liquid',
    )

    ax.set_title('Strike Price vs % Time Net Profit > 0 and % Time Both CE/PE Liquid')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Percentage of Time (%)')
    ax.set_ylim(0, 100)

    _set_strike_ticks(ax, summary['strike_price'])

    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    return summary[['strike_price', 'arbitrage_positive_pct', 'both_liquid_pct']]


def plot_strike_abs_violation_vs_cost_stripped_pct(df, strike):
    data = df[df['strike_price'] == strike].sort_values('fetch_time').copy()

    if data.empty:
        raise ValueError(f"No rows found for strike {strike}")

    # 1. Get the Magnitude of the Violation
    abs_violation = _get_abs_violation_series(data)

    # 2. Calculate Costs
    cost = data.apply(compute_cost, axis=1)
    
    # 3. Calculate % Stripped (with safety for division by zero)
    stripped_pct = np.where(abs_violation > 0, (cost / abs_violation) * 100.0, np.nan)
    
    # Cap for visualization purposes (250% = Cost is 2.5x the profit)
    display_pct = np.clip(stripped_pct, 0, 250) 

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    # Plot 1: Gross violation line turns green if cost strip < 100%, else red.
    x = data['fetch_time'].values
    x_num = date2num(x)
    if len(data) > 1:
        points = np.array([x_num, abs_violation]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = []
        for i in range(len(data) - 1):
            pct = stripped_pct[i]
            if np.isfinite(pct) and pct < 100:
                seg_colors.append('green')
            else:
                seg_colors.append('red')

        lc = LineCollection(segments, colors=seg_colors, linewidths=1.8)
        ax1.add_collection(lc)
        ax1.autoscale_view()
    else:
        single_color = 'green' if (np.isfinite(stripped_pct[0]) and stripped_pct[0] < 100) else 'red'
        ax1.plot(data['fetch_time'], abs_violation, color=single_color, linewidth=1.8)
    
    # Plot 2: The Cost Tax (Percentage)
    line2, = ax2.plot(data['fetch_time'], display_pct, color='tab:orange', 
                      linewidth=1.5, alpha=0.8, label='% Stripped by Cost')

    # Reference Lines
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=100, color='tab:red', linestyle='--', linewidth=2, alpha=0.6, label='Breakeven (Cost = Profit)')

    # Formatting X-Axis
    _apply_intraday_time_axis(ax1)

    # Labels and Limits
    ax1.set_title(f'Strike {strike}: Absolute Arbitrage Gap vs. Transaction Friction')
    ax1.set_xlabel('Time of Day (April 6th)')
    ax1.set_ylabel('Gross Violation Magnitude (₹)')
    ax2.set_ylabel('Cost as % of Violation')
    ax2.set_ylim(0, 250) 

    # Unified Legend
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Gross Violation (Cost Strip < 100%)'),
        Line2D([0], [0], color='red', lw=2, label='Gross Violation (Cost Strip >= 100%)'),
        line2,
    ]
    ax1.legend(legend_elements, [
        legend_elements[0].get_label(),
        legend_elements[1].get_label(),
        line2.get_label(),
    ], loc='upper right')
    ax1.grid(True, which='both', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        'fetch_time': data['fetch_time'],
        'abs_violation': abs_violation,
        'cost': cost,
        'cost_stripped_pct': stripped_pct,
    })


def plot_abs_violation_vs_net_profit(df, strike=None):
    data = df.copy()

    if strike is not None:
        data = data[data['strike_price'] == strike].copy()

    if data.empty:
        strike_msg = f" for strike {strike}" if strike is not None else ""
        raise ValueError(f"No rows found{strike_msg}")

    abs_violation = _get_abs_violation_series(data)

    if 'net_profit_per_unit' in data.columns:
        net_profit = data['net_profit_per_unit']
    else:
        net_profit = _get_net_profit_series(data, trading_cost=True, per_lot=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(abs_violation, net_profit, s=12, alpha=0.4, color='tab:blue')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.6)

    title = '|Violation| vs Net Arbitrage Profit'
    if strike is not None:
        title += f' (Strike {strike})'

    ax.set_title(title)
    ax.set_xlabel('|Violation|')
    ax.set_ylabel('Net Arbitrage Profit per Unit')
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        'abs_violation': abs_violation,
        'net_profit_per_unit': net_profit,
    })


def plot_moneyness_vs_net_profit(df, strike=None, min_liquidity=0):
    data = df.copy()

    if strike is not None:
        data = data[data['strike_price'] == strike].copy()

    if min_liquidity > 0:
        data = data[data['liquidity'] >= min_liquidity].copy()

    if data.empty:
        strike_msg = f" for strike {strike}" if strike is not None else ""
        raise ValueError(f"No rows found{strike_msg}")

    if 'spot_price' not in data.columns or 'strike_price' not in data.columns:
        raise KeyError("Expected both 'spot_price' and 'strike_price' columns")

    if 'net_profit_per_unit' in data.columns:
        net_profit = data['net_profit_per_unit']
    else:
        net_profit = _get_net_profit_series(data, trading_cost=True, per_lot=False)

    moneyness = data['spot_price'] / data['strike_price']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(moneyness, net_profit, s=12, alpha=0.45, c=net_profit, cmap='RdYlGn')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.6)
    ax.axvline(x=1.0, color='tab:blue', linestyle=':', alpha=0.7)

    title = 'Moneyness (S/K) vs Net Profit per Unit'
    if strike is not None:
        title += f' (Strike {strike})'

    ax.set_title(title)
    ax.set_xlabel('Moneyness (S/K)')
    ax.set_ylabel('Net Profit per Unit (₹)')
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        'moneyness': moneyness,
        'net_profit_per_unit': net_profit,
    })


def plot_option_chain_heatmap_over_time(
    df,
    min_liquidity=0,
    trading_cost=True,
    per_lot=False,
    figsize=(16, 10),
    efficiency_band_pct=0.03,
):
    """
    Create a comprehensive heatmap showing the entire option chain over the trading day.
    
    X-axis: Time of Day (09:15 to 15:30)
    Y-axis: Strike Prices (sorted)
    Color (Z-axis): Net Profit (Gross Violation - Total Cost)
    
    Green: Positive profit, Red: Negative (loss), White: Efficiency Zone (near break-even)
    
    Parameters:
    -----------
    df : DataFrame
        The full option chain data with fetch_time, strike_price, violation, etc.
    min_liquidity : int
        Minimum bottleneck OI threshold to include data (default: 0 = include all)
    trading_cost : bool
        If True, net profit includes trading costs. If False, shows raw violation.
    figsize : tuple
        Figure size (width, height)
    """
    
    # Filter for sufficient liquidity if specified
    if min_liquidity > 0:
        data = df[df['liquidity'] >= min_liquidity].copy()
    else:
        data = df.copy()
    
    if data.empty:
        print("No data available after filtering")
        return
    
    data['profit_value'] = _get_net_profit_series(
        data,
        trading_cost=trading_cost,
        per_lot=per_lot,
    )
    
    # Create pivot table: rows=strikes, columns=times, values=net_profit
    pivot_data = data.pivot_table(
        index='strike_price',
        columns='fetch_time',
        values='profit_value',
        aggfunc='mean'  # In case of duplicates, take mean
    )
    
    # Sort strikes from lowest to highest
    pivot_data = pivot_data.sort_index()
    
    # Extract time labels (format HH:MM)
    time_labels = [t.strftime('%H:%M') for t in pivot_data.columns]
    
    # Determine vmin, vmax for centered colormap
    abs_max = max(abs(pivot_data.min().min()), abs(pivot_data.max().max()))
    vmin = -abs_max
    vmax = abs_max

    # Small efficiency zone around zero with rapid but smooth transition outside it.
    efficiency_band = max(abs_max * efficiency_band_pct, 1e-6)
    norm = mcolors.SymLogNorm(
        linthresh=efficiency_band,
        linscale=0.25,
        vmin=vmin,
        vmax=vmax,
        base=10,
    )
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Rich multi-stop diverging map: red (loss) -> white (efficiency zone) -> green (profit).
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'red_white_green_rich',
        [
            '#8b0000',  # deep red
            '#d73027',
            '#f46d43',
            '#fdae61',
            '#fee08b',
            '#ffffff',  # center efficiency zone
            '#d9ef8b',
            '#a6d96a',
            '#66bd63',
            '#1a9850',
            '#006837',  # deep green
        ],
        N=1024,
    )

    # Smooth rendering with rapid color transition around the efficiency boundary.
    im = ax.imshow(
        pivot_data.values,
        cmap=cmap,
        aspect='auto',
        norm=norm,
        interpolation='nearest'
    )
    
    # Set Y-axis (strikes)
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index.astype(int))
    ax.set_ylabel('Strike Price', fontsize=12, fontweight='bold')
    
    # Set X-axis (time)
    # Sample every nth time for readable labels
    n_times = len(pivot_data.columns)
    step = max(1, n_times // 20)  # Show ~20 labels max
    time_tick_positions = range(0, n_times, step)
    time_tick_labels = [time_labels[i] for i in time_tick_positions]
    ax.set_xticks(time_tick_positions)
    ax.set_xticklabels(time_tick_labels, rotation=45, ha='right')
    ax.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
    
    # Add colorbar
    unit_label = 'Lot' if per_lot else 'Unit'
    plt.colorbar(im, ax=ax, label=f'Net Profit per {unit_label} (₹)', pad=0.02)
    
    # Title and labels
    title = 'Put-Call Parity Arbitrage: Option Chain Heatmap (Entire Trading Day)'
    if min_liquidity > 0:
        title += f'\n(Min Liquidity: {min_liquidity} contracts)'
    if trading_cost:
        title += f'\nNet Profit per {unit_label} includes Trading Costs'
    else:
        title += f'\nGross Violation Magnitude per {unit_label}'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for readability
    ax.set_xticks([i - 0.5 for i in range(1, n_times)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, len(pivot_data.index))], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pivot_data


def plot_option_chain_heatmap_statistics(df, min_liquidity=0):
    """
    Generate summary statistics from the heatmap data to identify deviations.
    
    Returns:
    --------
    summary : DataFrame
        Per-strike statistics including:
        - mean_profit: Average net profit across the day
        - max_profit: Maximum net profit observed
        - min_profit: Minimum net profit (most negative)
        - profitable_pct: Percentage of time the strike was profitable
        - liquidity_pct: Percentage of time both legs were liquid
    """
    
    if 'net_profit_per_unit' not in df.columns:
        df = df.copy()
        df['net_profit_per_unit'] = _get_net_profit_series(df, trading_cost=True, per_lot=False)
    
    if min_liquidity > 0:
        data = df[df['liquidity'] >= min_liquidity].copy()
    else:
        data = df.copy()
    
    summary = data.groupby('strike_price').agg({
        'net_profit_per_unit': ['mean', 'max', 'min', lambda x: (x > 0).mean() * 100],
        'both_legs_liquid': 'mean',
        'liquidity': 'mean'
    }).round(2)
    
    # Flatten column names
    summary.columns = [
        'Mean_Profit',
        'Max_Profit',
        'Min_Profit',
        'Profitable_%',
        'Liquid_Time_%',
        'Avg_Liquidity'
    ]
    
    summary = summary.sort_values('Mean_Profit', ascending=False)
    
    print("\n" + "="*80)
    print("ARBITRAGE OPPORTUNITY SUMMARY BY STRIKE")
    print("="*80)
    print(summary.to_string())
    print("="*80)
    
    return summary