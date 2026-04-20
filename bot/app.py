import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import utils
import config

# --- HELPER FUNCTIONS ---

def _extract_next_data(html: str) -> dict:
    """Extracts the JSON payload from the Groww page HTML."""
    marker = '<script id="__NEXT_DATA__"'
    start = html.find(marker)
    if start == -1: return {}
    tag_end = html.find(">", start)
    if tag_end == -1: return {}
    script_end = html.find("</script>", tag_end)
    if script_end == -1: return {}
    payload = html[tag_end + 1 : script_end].strip()
    return json.loads(payload)

def _flatten_dict(prefix: str, value: any) -> dict:
    """Safely flattens nested dictionaries from the API response."""
    flat = {}
    if not isinstance(value, dict):
        return flat
    for key, item in value.items():
        col = f"{prefix}_{key}"
        if isinstance(item, dict):
            flat.update(_flatten_dict(col, item))
        else:
            flat[col] = item
    return flat

def fetch_data_standalone(symbol: str, expiry: str):
    """Standalone fetcher to get data."""
    clean_symbol = str(symbol).lower()
    url = f"https://groww.in/options/{clean_symbol}?expiry={expiry}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        next_data = _extract_next_data(response.text)
        
        data = next_data.get("props", {}).get("pageProps", {}).get("data", {})
        spot_price = data.get("company", {}).get("liveData", {}).get("ltp")
        contracts = data.get("optionChain", {}).get("optionContracts", [])
        
        if not contracts: return pd.DataFrame()

        rows = []
        for contract in contracts:
            if not isinstance(contract, dict): continue
            row = {
                "symbol": symbol.upper(),
                "expiry": expiry,
                "spot_price": spot_price,
                "strike_price": contract.get("strikePrice", 0) / 100,
                "fetch_time": datetime.now() 
            }
            for side in ["ce", "pe"]:
                side_data = contract.get(side)
                if isinstance(side_data, dict):
                    row.update(_flatten_dict(side, side_data))
                else:
                    row[f"{side}_liveData_ltp"] = np.nan
                    row[f"{side}_liveData_oi"] = 0
                    row[f"{side}_markers"] = ""
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Fetch Error: {e}")
        return pd.DataFrame()

# --- STREAMLIT UI ---

st.set_page_config(page_title="NIFTY Arbitrage Radar", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    .strike-label { font-size: 24px; font-weight: bold; color: #00d4ff; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR & SESSION STATE ---

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Pause Toggle
    is_paused = st.checkbox("⏸️ Pause Auto-Refresh", value=False)
    
    symbol = "NIFTY"
    st.write(f"**Target:** {symbol}")
    
    expiry = st.text_input("Expiry (YYYY-MM-DD)", value="2026-04-28")
    
    st.divider()
    st.subheader("Config Overrides")
    config.tr = st.number_input("Brokerage %", value=config.tr, format="%.5f")
    config.B = st.number_input("Brokerage Cap", value=config.B)
    config.rl = st.number_input("Lending Rate", value=config.rl)
    config.rb = st.number_input("Borrowing Rate", value=config.rb)
    config.N = st.number_input("Lot Size", value=config.N)
    
    if is_paused:
        st.warning("Auto-refresh paused. Change params or press Enter to update manually.")
    else:
        st.success("Auto-refreshing every 30 seconds.")

# --- DASHBOARD LOGIC ---

# Only refresh every 30s if NOT paused
refresh_interval = None if is_paused else 30

@st.fragment(run_every=refresh_interval)
def dashboard_body():
    raw_df = fetch_data_standalone(symbol, expiry)
    
    if raw_df.empty:
        st.warning("Waiting for data or invalid Expiry...")
        return

    df = utils.load_and_clean_data(raw_df)
    df = utils.identify_arbitrage(df)
    df = utils.apply_costs(df)
    
    # Calculations
    df['arb_pct_spot'] = (df['violation'] / df['spot_price']) * 100
    max_oi = df[['ce_liveData_oi', 'pe_liveData_oi']].max().max()

    # Header
    col_spot, col_time, col_status = st.columns([1, 1, 1])
    col_spot.metric("NIFTY Spot", f"₹{df['spot_price'].iloc[0]:,.2f}")
    col_time.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    col_status.metric("Mode", "⏸️ Paused" if is_paused else "🔄 Live")

    st.divider()

    # Grid Headers
    h1, h2, h3 = st.columns([1, 2, 2])
    h1.write("**STRIKE**")
    h2.write("**ARB % OF SPOT**")
    h3.write("**LIQUIDITY (CE Blue | PE Orange)**")

    for _, row in df.sort_values('strike_price').iterrows():
        c1, c2, c3 = st.columns([1, 2, 2])
        
        # 1. Strike Price
        c1.markdown(f"<div class='strike-label'>{int(row['strike_price'])}</div>", unsafe_allow_html=True)
        
        # 2. Arbitrage Horizontal Histogram
        fig_arb = go.Figure(go.Bar(
            x=[row['arb_pct_spot']],
            orientation='h',
            marker_color='rgba(0, 255, 128, 0.8)' if row['arb_pct_spot'] > 0 else 'rgba(255, 75, 75, 0.8)',
            text=f"{row['violation']:.2f} ({row['arb_pct_spot']:.4f}%)",
            textposition='auto',
        ))
        fig_arb.update_layout(
            xaxis=dict(range=[-0.1, 0.1], visible=False), 
            yaxis=dict(visible=False),
            margin=dict(l=0, r=5, t=5, b=5), height=50,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        c2.plotly_chart(fig_arb, use_container_width=True, config={'displayModeBar': False}, key=f"arb_{row['strike_price']}")

        # 3. Liquidity Metrics
        fig_liq = go.Figure()
        fig_liq.add_trace(go.Bar(
            x=[row['ce_liveData_oi']], name="CE", orientation='h', marker_color='#00d4ff'
        ))
        fig_liq.add_trace(go.Bar(
            x=[row['pe_liveData_oi']], name="PE", orientation='h', marker_color='#ff8c00'
        ))
        fig_liq.update_layout(
            barmode='group',
            xaxis=dict(range=[0, max_oi], visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=5, t=5, b=5), height=50,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        c3.plotly_chart(fig_liq, use_container_width=True, config={'displayModeBar': False}, key=f"liq_{row['strike_price']}")


dashboard_body()