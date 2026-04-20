import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse

import pandas as pd
import numpy as np
import requests

def _extract_next_data(html: str) -> dict[str, Any]:
    """Extracts the JSON payload from the Groww page HTML."""
    marker = '<script id="__NEXT_DATA__"'
    start = html.find(marker)
    if start == -1:
        raise ValueError("Could not find __NEXT_DATA__ block in Groww page HTML.")

    tag_end = html.find(">", start)
    if tag_end == -1:
        raise ValueError("Malformed __NEXT_DATA__ script tag in Groww page HTML.")

    script_end = html.find("</script>", tag_end)
    if script_end == -1:
        # Final attempt: handle cases where script might end differently
        script_end = html.find(";", tag_end)
        if script_end == -1:
            raise ValueError("Could not find closing script tag for __NEXT_DATA__.")

    payload = html[tag_end + 1 : script_end].strip()
    return json.loads(payload)

def _flatten_dict(prefix: str, value: Any) -> dict[str, Any]:
    """
    Safely flattens nested dictionaries. 
    Prevents 'method is not iterable' by checking if the value is a dict.
    """
    flat: dict[str, Any] = {}
    if not isinstance(value, dict):
        return flat
        
    for key, item in value.items():
        col = f"{prefix}_{key}"
        if isinstance(item, dict):
            flat.update(_flatten_dict(col, item))
        else:
            flat[col] = item
    return flat

def fetch_groww_option_chain(symbol: str, expiry: str) -> pd.DataFrame:
    """Fetches and processes live option chain data from Groww."""
    clean_symbol = str(symbol).lower()
    url = f"https://groww.in/options/{clean_symbol}?expiry={expiry}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        next_data = _extract_next_data(response.text)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

    page_props = next_data.get("props", {}).get("pageProps", {})
    data = page_props.get("data", {})
    
    # Extract spot price
    live_data = data.get("company", {}).get("liveData", {})
    spot_price = live_data.get("ltp")
    
    # Extract contracts
    option_chain = data.get("optionChain", {})
    contracts = option_chain.get("optionContracts", [])

    if not contracts:
        return pd.DataFrame()

    rows = []
    for contract in contracts:
        if not isinstance(contract, dict):
            continue
            
        row: dict[str, Any] = {
            "symbol": symbol.upper(),
            "expiry": expiry,
            "spot_price": spot_price,
            "strike_price": contract.get("strikePrice", 0) / 100,
        }
        
        # Process Call (ce) and Put (pe) safely
        for side in ["ce", "pe"]:
            side_data = contract.get(side)
            if isinstance(side_data, dict):
                # Flattens nested keys like liveData_ltp, greeks_delta, etc.
                row.update(_flatten_dict(side, side_data))
            else:
                # Add default values for missing sides to prevent downstream KeyErrors
                row[f"{side}_liveData_ltp"] = np.nan
                row[f"{side}_liveData_oi"] = 0
                row[f"{side}_markers"] = "[]"

        rows.append(row)
        
    return pd.DataFrame(rows)

def build_ltp_view(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a compact view of the option chain."""
    cols = ["spot_price", "ce_liveData_ltp", "strike_price", "pe_liveData_ltp"]
    # Filter only available columns
    existing_cols = [c for c in cols if c in df.columns]
    
    return df.reindex(columns=existing_cols).rename(
        columns={
            "spot_price": "Spot Price",
            "ce_liveData_ltp": "Call LTP",
            "strike_price": "Strike Price",
            "pe_liveData_ltp": "Put LTP",
        }
    )

def append_csv(df: pd.DataFrame, output_file: Path) -> None:
    """Appends data to a CSV file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_file.exists()
    df.to_csv(output_file, mode="a", header=write_header, index=False)

def collect_continuously(symbol: str, expiry: str, interval_seconds: float, max_iterations: int | None) -> None:
    """Main loop for historical data collection."""
    today = datetime.now().strftime("%Y-%m-%d")
    base_dir = Path("data") / today
    full_history_file = base_dir / f"option_chain_{symbol.lower()}_{expiry}_full_history.csv"

    print(f"Collecting {symbol} data every {interval_seconds}s. Ctrl+C to stop.")

    iteration = 0
    while True:
        started = time.time()
        fetch_time = datetime.now().isoformat(timespec="seconds")

        df = fetch_groww_option_chain(symbol, expiry)

        if not df.empty:
            df.insert(0, "fetch_time", fetch_time)
            append_csv(df, full_history_file)
            print(f"[{fetch_time}] Saved {len(df)} rows")
        else:
            print(f"[{fetch_time}] No data found")

        iteration += 1
        if max_iterations is not None and iteration >= max_iterations:
            break

        elapsed = time.time() - started
        time.sleep(max(0.1, interval_seconds - elapsed))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Groww option-chain collector")
    parser.add_argument("--symbol", default="nifty", help="Symbol, e.g., nifty")
    parser.add_argument("--expiry", default="2026-04-28", help="Expiry YYYY-MM-DD")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval in seconds")
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop after N fetches")
    parser.add_argument("--once", action="store_true", help="Single fetch only")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.once:
        df = fetch_groww_option_chain(args.symbol, args.expiry)
        if not df.empty:
            print(df.head())
            df.to_csv(f"{args.symbol}_{args.expiry}.csv", index=False)
    else:
        collect_continuously(args.symbol, args.expiry, args.interval, args.max_iterations)