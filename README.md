# Schwab Portfolio Order Generator

A Python utility that processes Citrindex portfolio Excel files and generates Schwab-compatible order files.

## Features

- Parses Citrindex portfolio Excel files, attempting to find the correct header row automatically.
- Calculates target allocations based on **Target Gross Exposure** (Sum of Long Market Values + Absolute Sum of Short Market Values).
- Handles both long and short positions using specified weights (e.g., "Wgt", "Delta Adj Wgt", "Market Value (%)").
- Normalizes weights based on the *absolute* value of weights in the (potentially filtered) target portfolio.
- Compares target portfolio against a **current portfolio CSV** (`Individual-Positions.csv` format) to generate net orders.
- Classifies orders as "New", "Adjust", or "Exit".
- Sorts the final order list to keep the original Citrindex sequence for "New" and "Adjust" orders, with "Exit" orders placed at the bottom.
- Optionally filters for US-only stocks (`--us-only`).
- Optionally attempts to find US-equivalent tickers (ADRs, common mappings) for non-US stocks (`--find-us-replacements`).
- Optionally fetches updated market prices via yfinance (`--update-prices`).
- Handles ticker normalization (removes suffixes, standardizes formats).
- Generates multiple output formats in an `orders/` subdirectory:
  - Detailed net orders in Excel (`schwab_orders.xlsx` by default) including the `Order Type`.
  - Detailed net orders in CSV (`net_orders_detailed.csv`).
  - Simplified Schwab-compatible upload CSV (`schwab_upload.csv`).

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/philippe-eecs/schwab.git
cd schwab
pip install -r requirements.txt
```
*(Ensure `requirements.txt` includes pandas, numpy, requests, beautifulsoup4, yfinance, and openpyxl)*

## Usage

### Basic Usage

Process a Citrindex file against a current holdings CSV, specifying target gross exposure:

```bash
python process_citrini.py "path/to/Citrindex File.xlsx" --current-portfolio-csv "path/to/Individual-Positions.csv" --target-gross-exposure 500000
```

### Command Line Options

```
usage: process_citrini.py [-h] [--target-gross-exposure TARGET_GROSS_EXPOSURE]
                          [--output-file OUTPUT_FILE] [--min-allocation MIN_ALLOCATION]
                          [--round-up | --no-round-up] [--use-delta] [--us-only]
                          [--find-us-replacements] [--update-prices]
                          [--current-portfolio-csv CURRENT_PORTFOLIO_CSV]
                          [--exclude EXCLUDE [EXCLUDE ...]]
                          input_file

Process Citrindex portfolio, compare with current holdings, and generate net Schwab orders

positional arguments:
  input_file            Path to Citrindex portfolio Excel file

options:
  -h, --help            show this help message and exit
  --target-gross-exposure TARGET_GROSS_EXPOSURE
                        Target Gross Exposure (Longs + Abs(Shorts)) in USD (default: 350000)
  --output-file OUTPUT_FILE
                        Base name for output Excel file (will be saved in ./orders/) (default: schwab_orders.xlsx)
  --min-allocation MIN_ALLOCATION
                        Minimum target allocation in USD to consider a position (default: 10)
  --round-up            Round up share quantities (default: True)
  --no-round-up         Round down share quantities instead of up
  --use-delta           Use Delta Adj Wgt if available (default: False, uses Wgt or Market Value %%)
  --us-only             Filter out non-US stocks *after* attempting replacements (default: False)
  --find-us-replacements
                        Try to find US replacements (ADRs/mapped tickers) for non-US stocks (default: False)
  --update-prices       Fetch current market prices using yfinance (default: False)
  --current-portfolio-csv CURRENT_PORTFOLIO_CSV
                        Path to Schwab positions CSV file (e.g., Individual-Positions.csv) for calculating net orders
  --exclude EXCLUDE [EXCLUDE ...]]
                        List of tickers to exclude from the target portfolio (e.g., --exclude CASH BRK.B)

```

### Examples

Allocate $1,000,000 Gross Exposure, comparing with current holdings, filtering for US only:

```bash
python process_citrini.py "Citrindex Report.xlsx" --current-portfolio-csv "Individual-Positions.csv" --target-gross-exposure 1000000 --us-only
```

Process without comparing to current holdings (generate target orders directly), use delta weights, and fetch current prices:

```bash
python process_citrini.py "Delta Adjusted Portfolio.xlsx" --target-gross-exposure 250000 --use-delta --update-prices
```

Include non-US stocks and try to find US replacements:

```bash
python process_citrini.py Citrindex_Global.xlsx --current-portfolio-csv Holdings.csv --target-gross-exposure 750000 --find-us-replacements
```


## Output Files

The script generates the following files in the `orders/` directory:

- **`[output-file-base].xlsx`** (e.g., `schwab_orders.xlsx`): Detailed net orders, including target vs. current quantities, calculated values, weights, the `Order Type` (New/Adjust/Exit), and final `Action` (BUY/SELL/etc.). Sorted by Order Type (Exits last), then by original Citrindex order.
- **`net_orders_detailed.csv`**: The same detailed information as the Excel file, but in CSV format.
- **`schwab_upload.csv`**: A simplified CSV format containing only `Symbol`, `Action` (mapped to Schwab terms like Buy/Sell/Sell Short), and `Quantity`, intended for easier upload or processing by Schwab systems.

## Weight Calculation Logic

The script calculates target allocations based on the `--target-gross-exposure` value.
1. It identifies the appropriate weight column in the input file ("Wgt", "Delta Adj Wgt", or "Market Value (%)").
2. After initial filtering (e.g., `--us-only`, `--exclude`), it calculates the sum of the *absolute* values of the weights for the remaining tickers.
3. It calculates a `NormWeight` for each ticker by dividing its `Raw Weight` by the total absolute weight sum.
4. The target dollar allocation for each ticker is calculated as `Target USD = NormWeight * target_gross_exposure`.

This means the `target_gross_exposure` is distributed across the selected tickers according to their relative absolute weights. The resulting *net* exposure (Longs - Shorts) will depend on the specific weights in the filtered portfolio.

## Requirements

- Python 3.6+
- pandas
- numpy
- requests
- beautifulsoup4
- yfinance
- openpyxl (for reading/writing Excel files)

## License

MIT
