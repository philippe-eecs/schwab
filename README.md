# Schwab Portfolio Order Generator

A Python utility that processes Citrindex portfolio Excel files and generates Schwab-compatible order files.

## Features

- Parses Citrindex portfolio Excel files with accurate weight handling
- Supports both long and short positions
- Properly calculates dollar allocations based on target net investment
- Rounds shares to the nearest whole number, so won't be perfect in hitting target
- I haven't setup the options properly so I am not sure if its done right
- Generates multiple output formats:
  - Full order details in Excel format
  - Stocks orders CSV
  - Options orders CSV
  - Net orders CSV
  - Schwab-compatible upload CSV (I think)

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/philippe-eecs/schwab.git
cd schwab
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process a portfolio file with default settings:

```bash
python process_citrini.py path/to/portfolio.xlsx
```

### Command Line Options

The script supports several options:

```
python process_citrini.py portfolio.xlsx [options]
```

Options:
- `--net-investment AMOUNT`: Total capital to allocate (long - short). Default: 1,000,000
- `--output-file FILENAME`: Output Excel file. Default: "schwab_orders.xlsx"
- `--min-allocation AMOUNT`: Minimum allocation in USD. Default: 0
- `--round-up`: Round up share quantities (default: True)
- `--use-delta`: Use Delta Adj Wgt column if available (default: False)
- `--us-only`: Filter out non-US stocks (default: True)
- `--no-us-only`: Include non-US stocks
- `--find-us-replacements`: Try to find US replacements for non-US stocks (default: False)
- `--update-prices`: Fetch current market prices for tickers (default: False)
- `--current-portfolio FILE`: Path to Excel file with current holdings for calculating net orders

### Examples

Allocate $5 million net investment with a minimum allocation of $10,000:

```bash
python process_citrini.py portfolio.xlsx --net-investment 5000000 --min-allocation 10000
```

Include non-US stocks and try to find US replacements:

```bash
python process_citrini.py portfolio.xlsx --no-us-only --find-us-replacements
```

Calculate net orders by comparing with current holdings:

```bash
python process_citrini.py portfolio.xlsx --current-portfolio current_holdings.xlsx
```

## Output Files

The script generates the following files in the `orders/` directory:

- `stocks_orders.csv`: All stock orders with detailed information
- `options_orders.csv`: All options orders with detailed information
- `net_orders.csv`: Net orders (comparing with current portfolio if provided)
- `schwab_upload.csv`: Simplified format for upload to Schwab

## Weight Calculation Logic

The script applies weights directly to the net investment amount:
- A position with weight of +5% will be allocated 5% of the net investment amount for a long position
- A position with weight of -6% will be allocated -6% of the net investment amount for a short position

This ensures that the dollar allocations correctly reflect the specified weights relative to the total portfolio size.

## Requirements

- Python 3.6+
- pandas
- numpy
- requests
- beautifulsoup4
- yfinance

## License

MIT
