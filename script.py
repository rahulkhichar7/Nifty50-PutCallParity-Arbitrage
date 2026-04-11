import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse

import pandas as pd
import requests


def _extract_next_data(html: str) -> dict[str, Any]:
	marker = '<script id="__NEXT_DATA__"'
	start = html.find(marker)
	if start == -1:
		raise ValueError("Could not find __NEXT_DATA__ block in Groww page HTML.")

	tag_end = html.find(">", start)
	if tag_end == -1:
		raise ValueError("Malformed __NEXT_DATA__ script tag in Groww page HTML.")

	script_end = html.find("</script>", tag_end)
	if script_end == -1:
		raise ValueError("Could not find closing script tag for __NEXT_DATA__.")

	payload = html[tag_end + 1 : script_end].strip()
	return json.loads(payload)


def _flatten_dict(prefix: str, value: dict[str, Any]) -> dict[str, Any]:
	flat: dict[str, Any] = {}
	for key, item in value.items():
		col = f"{prefix}_{key}"
		if isinstance(item, dict):
			flat.update(_flatten_dict(col, item))
		else:
			flat[col] = item
	return flat


def fetch_groww_option_chain(symbol: str, expiry: str) -> pd.DataFrame:
	url = f"https://groww.in/options/{symbol.lower()}?expiry={expiry}"
	response = requests.get(url, timeout=30)
	response.raise_for_status()

	next_data = _extract_next_data(response.text)
	contracts = (
		next_data.get("props", {})
		.get("pageProps", {})
		.get("data", {})
		.get("optionChain", {})
		.get("optionContracts", [])
	)
	data = next_data.get("props", {}).get("pageProps", {}).get("data", {})
	spot_price = data.get("company", {}).get("liveData", {}).get("ltp")

	if not contracts:
		return pd.DataFrame()

	rows = []
	for contract in contracts:
		row: dict[str, Any] = {
			"symbol": symbol.upper(),
			"expiry": expiry,
			"spot_price": spot_price,
			"strike_price": contract.get("strikePrice", 0) / 100,
		}
		ce = contract.get("ce")
		pe = contract.get("pe")
		if isinstance(ce, dict):
			row.update(_flatten_dict("ce", ce))
		if isinstance(pe, dict):
			row.update(_flatten_dict("pe", pe))

		rows.append(row)
	return pd.DataFrame(rows)


def build_ltp_view(df: pd.DataFrame) -> pd.DataFrame:
	return df.reindex(columns=["spot_price", "ce_liveData_ltp", "strike_price", "pe_liveData_ltp"]).rename(
		columns={
			"spot_price": "Spot Price",
			"ce_liveData_ltp": "Call LTP",
			"strike_price": "Strike Price",
			"pe_liveData_ltp": "Put LTP",
		}
	)


def append_csv(df: pd.DataFrame, output_file: Path) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	write_header = not output_file.exists()
	df.to_csv(output_file, mode="a", header=write_header, index=False)


def collect_continuously(symbol: str, expiry: str, interval_seconds: float, max_iterations: int | None) -> None:
	today = datetime.now().strftime("%Y-%m-%d")
	base_dir = Path("data") / today
	full_history_file = base_dir / f"option_chain_{symbol.lower()}_{expiry}_full_history.csv"
	ltp_history_file = base_dir / f"option_chain_{symbol.lower()}_{expiry}_ltp_history.csv"

	print(f"Collecting live option chain every {interval_seconds} seconds.")
	print(f"Full history file: {full_history_file}")
	print(f"LTP history file: {ltp_history_file}")
	print("Press Ctrl+C to stop.")

	iteration = 0
	while True:
		started = time.time()
		fetch_time = datetime.now().isoformat(timespec="seconds")

		try:
			df = fetch_groww_option_chain(symbol, expiry)
		except Exception as exc:
			print(f"[{fetch_time}] Fetch failed: {exc}")
			df = pd.DataFrame()

		if not df.empty:
			df.insert(0, "fetch_time", fetch_time)
			append_csv(df, full_history_file)

			ltp_df = build_ltp_view(df)
			ltp_df.insert(0, "fetch_time", fetch_time)
			append_csv(ltp_df, ltp_history_file)
			print(f"[{fetch_time}] Saved {len(df)} rows")
		else:
			print(f"[{fetch_time}] No contracts found")

		iteration += 1
		if max_iterations is not None and iteration >= max_iterations:
			break

		elapsed = time.time() - started
		time.sleep(max(0.0, interval_seconds - elapsed))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Groww option-chain collector")
	parser.add_argument("--symbol", default="nifty", help="Underlying symbol, e.g., nifty")
	parser.add_argument("--expiry", default="2026-04-28", help="Expiry date in YYYY-MM-DD")
	parser.add_argument(
		"--interval",
		type=float,
		default=2.0,
		help="Polling interval in seconds (default: 2.0)",
	)
	parser.add_argument(
		"--max-iterations",
		type=int,
		default=None,
		help="Stop after N fetches (default: run forever)",
	)
	parser.add_argument(
		"--once",
		action="store_true",
		help="Run a single fetch/export only",
	)
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	if args.once:
		df = fetch_groww_option_chain(args.symbol, args.expiry)
		output_file = f"option_chain_{args.symbol.lower()}_{args.expiry}.csv"
		df.to_csv(output_file, index=False)

		compact_output_file = f"option_chain_{args.symbol.lower()}_{args.expiry}_ltp_only.csv"
		compact_df = build_ltp_view(df)
		compact_df.to_csv(compact_output_file, index=False)

		print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
		print(f"Saved CSV: {output_file}")
		print(f"Saved CSV: {compact_output_file}")
		if not df.empty:
			print(df.head(10).to_string(index=False))
	else:
		collect_continuously(args.symbol, args.expiry, args.interval, args.max_iterations)

# To run
# python .\script.py --symbol nifty --expiry 2026-04-07