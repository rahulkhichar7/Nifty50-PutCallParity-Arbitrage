# Real-Time Detection of Put-Call Parity Arbitrage in Options Markets

## Overview

This project detects and analyzes **Put-Call Parity arbitrage opportunities** in real-time for Nifty50 options. It fetches live option chain data, identifies mispricings, calculates trading costs, and visualizes arbitrage opportunities with liquidity metrics.

### What is Put-Call Parity?

Put-Call Parity establishes the relationship between European call and put options:

**C - P = S - K×e^(-r×T)**

Where:
- C = Call option price
- P = Put option price  
- S = Current stock/index price
- K = Strike price
- r = Risk-free interest rate
- T = Time to expiration

When market prices deviate from this relationship, arbitrage opportunities exist.

## Repo Structure

```
.
├── script.py              # Main data collection script (real-time fetching)
├── utils.py              # Core functions for arbitrage detection & visualization
├── Test.ipynb            # Jupyter notebook for analysis and testing
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── Project Statements.pdf # Detailed project documentation
```

## How It Works

### 1. **Data Collection** (`script.py`)
- Fetches real-time option chain data from Groww for Nifty50
- Extracts Call and Put prices, strike prices, open interest, and liquidity markers
- Stores data in CSV format with timestamps

### 2. **Arbitrage Detection** (`utils.py`)
- **Spread Calculation**: Market-side vs Theoretical-side
- **Cost Analysis**: Includes brokerage, STT, transaction fees, GST, SEBI charges
- **Liquidity Filtering**: Identifies and flags illiquid contracts
- **Market Interest Rate**: Derives implied interest rates from market prices

### 3. **Visualization**
- Color-coded plots showing arbitrage spreads over time
- Green = Liquid contracts
- Yellow = Below liquidity threshold (OI < 500)
- Red = Explicitly marked as illiquid

## Installation & Setup

### Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation
```bash
pip install -r requirements.txt
```

### Dependencies
- pandas, numpy - Data processing
- requests - HTTP requests for data fetching
- matplotlib - Visualization
- jupyter - For running the analysis notebook

## Usage

### Real-Time Data Collection
```bash
python script.py --symbol NIFTY50 --expiry 2024-04-18 --interval 5 --max-iterations 100
```

Parameters:
- `--symbol`: Stock/Index symbol (e.g., NIFTY50)
- `--expiry`: Expiry date (YYYY-MM-DD)
- `--interval`: Collection interval in seconds
- `--max-iterations`: Number of data points to collect

### Analysis
See `Test.ipynb` for complete analysis workflow:
- Load and clean data
- Identify arbitrage opportunities
- Calculate profitability after costs
- Generate visualizations

## Key Functions

**`fetch_groww_option_chain(symbol, expiry)`**
- Fetches live option chain data from Groww

**`identify_arbitrage(df, r=0.07, avoid_illiquid=False, oi_threshold=500)`**
- Detects arbitrage opportunities
- Calculates spreads and market-implied rates

**`compute_cost(row)`**
- Calculates total trading costs including:
  - Brokerage (₹20 per contract side × 3)
  - STT (0.15%)
  - Transaction fee (0.0355%)
  - SEBI fee (0.000001%)
  - Stamp duty (0.003%)
  - GST (18% on applicable fees)

**`plot_arbitrage(df, strike, avoid_illiquid=False, trading_cost=True)`**
- Visualizes arbitrage opportunities over time
- Color-codes by liquidity

## Trading Costs Reference

Based on Zerodha pricing:
- **Brokerage**: ₹20 per contract side
- **STT**: 0.15% (selling side)
- **Transaction Fee**: 0.0355%
- **SEBI Charges**: 0.000001%
- **Stamp Duty**: 0.003% (buying side only)
- **GST**: 18% on applicable fees

## Key Considerations

1. **Liquidity**: Only trade contracts with sufficient open interest (OI > 500 recommended)
2. **Execution Risk**: Real arbitrage requires near-simultaneous execution of all legs
3. **Transaction Costs**: Profit must exceed total trading costs to be viable
4. **Market Impact**: Large orders may face slippage
5. **Data Freshness**: Uses real-time data from Groww; accuracy depends on live market conditions

## References

- **Paper**: [Put-Call Parity and Dividend Yields - DVPCPT](https://www.actuariesindia.org/sites/default/files/2022-05/DVPCPT_Dheeraj_Sangeeta_Misra.PDF)
- **Trading Costs**: [Zerodha Brokerage Charges](https://zerodha.com/charges/#tab-equities)

## Disclaimer

This tool is for educational and research purposes only. Trading options involves substantial risk. Always conduct your own due diligence and consult with financial advisors before trading. Past performance and analysis do not guarantee future results.

## Assumptions
- No slippage/ exicution delay. To add it in cost, modify the spread_cost parameter in config.py 
- Best lending rates and borrowing rates(SBI) are taken but this may depend on your source of fund & civil score.