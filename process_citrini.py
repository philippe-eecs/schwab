import pandas as pd
import numpy as np
import math
import re
import requests
from bs4 import BeautifulSoup
import argparse
from datetime import datetime
import os
import time
import json
import yfinance as yf

def lookup_us_ticker(ticker):
    """
    Improved utility to guess a US ticker for foreign symbols.
    Includes better error handling and detailed logging.
    """
    # Extract just the base ticker without country code
    base_ticker = ticker.split()[0] if " " in ticker else ticker
    
    # Return early if it's likely already a US ticker
    if " US" in ticker or ticker.endswith(".US"):
        return ticker.split()[0]  # Return without the US suffix
    
    # Simple manual mapping for common stocks for faster lookups
    mapping = {
        "SONY": "SONY",
        "BABA": "BABA", 
        "9984": "SFTBY",  # SoftBank
        "7974": "NTDOY",  # Nintendo
        "005930": "SSNLF",  # Samsung
        "TCEHY": "TCEHY",  # Tencent
        "NSANY": "NSANY",  # Nissan
        "TM": "TM",       # Toyota
        "HMC": "HMC"      # Honda
    }
    
    if base_ticker in mapping:
        print(f"[DEBUG] Found {base_ticker} in manual mapping -> {mapping[base_ticker]}")
        return mapping[base_ticker]
    
    # Try multiple search queries for better results
    search_queries = [
        f"{base_ticker}+US+stock+ticker+NYSE+NASDAQ",
        f"{base_ticker}+ADR+stock",
        f"{base_ticker}+US+equivalent+ticker"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    for query in search_queries:
        url = f"https://www.google.com/search?q={query}"
        
        try:
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
            print(f"[DEBUG] Searching for US ticker with query: {query}")
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"[WARNING] Got status code {response.status_code} for {url}")
                continue
                
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.text.upper()
            
            # Try several patterns to find US tickers
            # Pattern for "NASDAQ: XXXX" or "NYSE: XXXX"
            exchange_pattern = re.search(r"(NASDAQ:\s*([A-Z0-9.]+)|NYSE:\s*([A-Z0-9.]+))", text)
            if exchange_pattern:
                result = exchange_pattern.group(2) if exchange_pattern.group(2) else exchange_pattern.group(3)
                print(f"[DEBUG] Found exchange listing pattern: {result}")
                return result
            
            # Try to find ticker in format like "SONY (SONY)" where one is likely the ticker
            parenthesis_pattern = re.search(r"([A-Z]{1,5}) \([A-Z]{1,5}\)", text)
            if parenthesis_pattern:
                result = parenthesis_pattern.group(1)
                print(f"[DEBUG] Found parenthesis pattern: {result}")
                return result
            
            # Look for ADR pattern
            adr_pattern = re.search(r"([A-Z]{1,5}) ADR", text)
            if adr_pattern:
                result = adr_pattern.group(1)
                print(f"[DEBUG] Found ADR pattern: {result}")
                return result
                
        except requests.exceptions.Timeout:
            print(f"[WARNING] Request timed out for {url}")
        except requests.exceptions.ConnectionError:
            print(f"[WARNING] Connection error for {url}")
        except Exception as e:
            print(f"[WARNING] Error in lookup_us_ticker for {ticker} with query {query}: {str(e)}")
    
    print(f"[DEBUG] Could not find US ticker for {ticker}, returning original ticker")
    return ticker

def process_combined_sheet(
    input_file,
    net_investment,
    output_file=None,
    exclude=None,
    use_delta=False,
    min_allocation=0,
    round_up=True,
    us_only=True,
    find_us_replacements=False,
    update_prices=False,
    current_portfolio=None
):
    """
    Reads the single Excel sheet that has columns:

    Ticker, Market Value (%), Px Close, Crncy, FX Cls, Wgt, Mkt Val,
    Market Value (Abs), Delta Adj Exp, Delta Adj Wgt, ISIN, Cost Date, Pos (Disp)

    and creates a unified portfolio DataFrame.

    Parameters
    ----------
    input_file: str
        Path to the Excel file
    net_investment: float
        The total capital to allocate (long - short).
    output_file: str or None
        If given, writes out the final parsed orders to Excel and CSV.
    exclude: list or None
        Tickers to skip entirely.
    use_delta: bool
        Whether to prefer 'Delta Adj Wgt' over 'Market Value (%)' if available.
    min_allocation: float
        Ignore positions whose final $ is below this.
    round_up: bool
        True => math.ceil() the share count, else math.floor().
    us_only: bool
        If True, filter out non-US stocks.
    find_us_replacements: bool
        If True, try to find US replacements for non-US stocks.
    update_prices: bool
        If True, fetch current market prices for tickers (especially for US replacements).
    current_portfolio: str or None
        Path to Excel file containing current Schwab portfolio for calculating net orders.
    """
    if exclude is None:
        exclude = []

    # 1) Load the Excel, find the correct sheet that includes 'Ticker' column, etc.
    xls = pd.ExcelFile(input_file)
    df = None
    chosen_sheet = None
    for sheet in xls.sheet_names:
        for header_row in range(0, 20):
            # read a small portion for a quick check
            temp_df = pd.read_excel(input_file, sheet_name=sheet, header=header_row)
            # Check columns
            needed = ["Ticker", "Px Close"]  # minimal
            if all(col in temp_df.columns for col in needed):
                df = temp_df
                chosen_sheet = sheet
                break
        if df is not None:
            break

    if df is None:
        raise ValueError("No sheet found matching the required columns (Ticker, Px Close, etc.).")

    print(f"[DEBUG] Found data in sheet '{chosen_sheet}'")
    print(f"[DEBUG] Columns found: {df.columns.tolist()}")
    print(f"[DEBUG] Sample data (first 3 rows):")
    pd.set_option('display.max_columns', None)
    print(df.head(3))
    print(f"[DEBUG] Total rows in input: {len(df)}")

    

    # 2) Clean the data
    # Drop rows where Ticker is NaN or blank
    df = df.dropna(subset=["Ticker"])
    df = df[df["Ticker"].astype(str).str.strip() != ""]
    print(f"[DEBUG] After removing blank tickers: {len(df)} rows")

    # Exclude summary rows by checking specific values in the first column or Unnamed: 0
    summary_keywords = [
        "CITRINDEX: CORE PORTFOLIO", 
        "AIRLINES BASKET", 
        "CHINESE EQUITY BARBELL", 
        "CITRINI SMID CAP PICKS", 
        "FISCAL PRIMACY NARROWED", 
        "GLOBAL AI BASKET", 
        "MEDTECH & HEALTHCARE INNOVATI", 
        "NAT GAS", 
        "RTHEDG",
        "TOTAL", 
        "SUMMARY", 
        "FOOTNOTE", 
        "GRAND", 
        "CASH", 
        "AGGREGATE"
    ]
    
    # Check both the first column and Unnamed: 0 for summary keywords
    first_col = df.columns[0]
    mask = df[first_col].astype(str).apply(lambda x: not any(keyword in x for keyword in summary_keywords))
    if "Unnamed: 0" in df.columns and first_col != "Unnamed: 0":
        mask = mask & df["Unnamed: 0"].astype(str).apply(lambda x: not any(keyword in x for keyword in summary_keywords))
    
    # Also check the Ticker column for summary keywords
    mask = mask & df["Ticker"].astype(str).apply(lambda x: not any(keyword in x for keyword in summary_keywords))
    
    df = df[mask]
    print(f"[DEBUG] After excluding summary rows: {len(df)} rows")

    # If user gave some tickers to exclude
    df = df[~df["Ticker"].isin(exclude)]
    print(f"[DEBUG] After excluding user-specified tickers: {len(df)} rows")

    # Also skip extremely large or nonsensical weights
    # We'll skip if we have 'Market Value (%)' or 'Delta Adj Wgt' columns that are beyond ±500
    if "Market Value (%)" in df.columns:
        # Fix the logical error - we want to keep rows where weights are within reasonable range
        df = df[~(df["Market Value (%)"].abs() > 500)]
        print(f"[DEBUG] After filtering extreme weights: {len(df)} rows")

    # Instead of the above, let's do a more direct approach:
    # We'll handle the numeric conversion carefull

    if "Market Value (%)" not in df.columns and "Delta Adj Wgt" not in df.columns and "Wgt" not in df.columns:
        raise ValueError("Neither 'Market Value (%)', 'Delta Adj Wgt', nor 'Wgt' columns found. Can't allocate weights.")

    # 3) Choose a weight column
    # If use_delta==True and 'Delta Adj Wgt' is present, prefer that.
    # else fall back to 'Market Value (%)' or 'Wgt'
    if use_delta and "Delta Adj Wgt" in df.columns:
        weight_col = "Delta Adj Wgt"
        print("[DEBUG] Using 'Delta Adj Wgt' as the weighting column.")
    else:
        weight_col = "Wgt"
        print("[DEBUG] Using 'Wgt' as the weighting column.")

    df["NormWeight"] = df[weight_col]

    # Check if weights sum to approximately 1
    total_weight = df["NormWeight"].sum()
    df["NormWeight"] = df["NormWeight"] / total_weight
    #Find total long weights and short weights in absolute terms and determine the amount of money to allocate to each
    total_long = df.loc[df["NormWeight"] > 0, "NormWeight"].sum()
    total_short = abs(df.loc[df["NormWeight"] < 0, "NormWeight"].sum())

    print(f"[DEBUG] Total long weights: {total_long:.4f}")
    print(f"[DEBUG] Total short weights: {total_short:.4f}")
    #print(f"[DEBUG] Sum of raw weights before normalization: {pre_norm_sum:.4f}")
    print(f"[DEBUG] Check sum AFTER normalization: {df['NormWeight'].sum():.4f}")

    # Apply weights directly to the net investment amount
    # Instead of scaling longs and shorts separately
    df["DollarAllocation"] = df["NormWeight"] * net_investment

    # Show the allocation for debugging
    allocated_long = df.loc[df["NormWeight"] > 0, "DollarAllocation"].sum()
    allocated_short = abs(df.loc[df["NormWeight"] < 0, "DollarAllocation"].sum())
    print(f"[DEBUG] Allocated to longs: ${allocated_long:,.2f}")
    print(f"[DEBUG] Allocated to shorts: ${allocated_short:,.2f}")
    print(f"[DEBUG] Net = {allocated_long - allocated_short:,.2f} (should be equal to {net_investment:,.2f})")

    # If user wants US stocks only, filter out non-US stocks
    if us_only:
        # Identify US stocks (typically have 'US' suffix in ticker or USD currency)
        us_stocks = df["Ticker"].str.endswith(" US") | (df["Crncy"] == "USD")
        non_us_count = (~us_stocks).sum()
        print(f"[DEBUG] Filtering out {non_us_count} non-US stocks")
        df = df[us_stocks]
        print(f"[DEBUG] After filtering for US stocks only: {len(df)} rows")
    
    # If user wants to find US replacements for non-US stocks
    if find_us_replacements and not us_only:
        # Identify non-US stocks
        non_us_stocks = ~(df["Ticker"].str.endswith(" US") | (df["Crncy"] == "USD"))
        non_us_tickers = df.loc[non_us_stocks, "Ticker"].tolist()
        
        print(f"[DEBUG] Finding US replacements for {len(non_us_tickers)} non-US stocks")
        
        # Try to find US replacements
        replacements_found = 0
        replacements = {}
        
        # Use rate limiting to avoid being blocked
        for i, ticker in enumerate(non_us_tickers):
            try:
                # Add rate limiting to prevent overloading the server
                if i > 0 and i % 5 == 0:
                    print(f"[DEBUG] Pausing for rate limiting after {i} lookups...")
                    time.sleep(2)
                
                us_ticker = lookup_us_ticker(ticker)
                if us_ticker != ticker:  # If a replacement was found
                    replacements[ticker] = us_ticker
                    # Update the ticker in the DataFrame
                    df.loc[df["Ticker"] == ticker, "Ticker"] = us_ticker
                    # Set currency to USD
                    df.loc[df["Ticker"] == us_ticker, "Crncy"] = "USD"
                    
                    # If update_prices is True, get current market price for the US ticker
                    if update_prices:
                        current_price, currency = fetch_current_price(us_ticker)
                        if current_price is not None:
                            print(f"[DEBUG] Updated price for {us_ticker}: ${current_price:.2f}")
                            df.loc[df["Ticker"] == us_ticker, "Px Close"] = current_price
                            # Keep FX rate as 1.0 for USD
                            df.loc[df["Ticker"] == us_ticker, "FX Cls"] = 1.0
                        else:
                            # If we couldn't get a price, keep FX rate as 1.0 for USD
                            df.loc[df["Ticker"] == us_ticker, "FX Cls"] = 1.0
                    else:
                        # If not updating prices, just set FX rate to 1.0 for USD
                        df.loc[df["Ticker"] == us_ticker, "FX Cls"] = 1.0
                    
                    replacements_found += 1
            except Exception as e:
                print(f"[DEBUG] Error finding replacement for {ticker}: {str(e)}")
                # Continue with next ticker rather than failing the whole process
                continue
        
        print(f"[DEBUG] Found US replacements for {replacements_found} non-US stocks")
        if replacements:
            print("[DEBUG] Replacements found:")
            for orig, repl in replacements.items():
                print(f"  {orig} -> {repl}")
    
    # Option to update prices for all tickers
    if update_prices:
        print("[DEBUG] Updating prices for all tickers...")
        # Get a list of all tickers, both US and non-US
        all_tickers = df["Ticker"].unique().tolist()
        
        prices_updated = 0
        for i, ticker in enumerate(all_tickers):
            # Add rate limiting
            if i > 0 and i % 10 == 0:
                print(f"[DEBUG] Pausing for rate limiting after {i} price lookups...")
                time.sleep(2)
            
            # Skip non-US tickers if replacements were already found
            if find_us_replacements and not us_only and ticker in replacements.values():
                continue
                
            # Only update prices for USD tickers or if us_only is False
            curr = df.loc[df["Ticker"] == ticker, "Crncy"].iloc[0]
            if curr == "USD" or not us_only:
                current_price, currency = fetch_current_price(ticker)
                if current_price is not None:
                    # Update the price
                    df.loc[df["Ticker"] == ticker, "Px Close"] = current_price
                    prices_updated += 1
        
        print(f"[DEBUG] Updated prices for {prices_updated} tickers")

    # 5) Now parse each row into final allocations
    # We'll skip re-normalization. We assume the sheet sums to about ±1.0 total
    # or the user is comfortable with how it's laid out.
    # We'll just scale by net_investment.

    # Some optional: parse currency, fx
    # Also see if it's an option by checking Ticker with e.g. [CP]
    option_pattern = re.compile(r"(?:\d{6}[CP]\d{8}|OPTION|OPT|/| [CP]\d{2,4}| P\d| C\d)")
    
    # First pass - calculate theoretical positions with fractional shares
    theoretical_positions = []
    
    for idx, row in df.iterrows():
        ticker_raw = str(row["Ticker"]).strip()
        w_float = row["NormWeight"]
        target_usd = row["DollarAllocation"]  # Use direct dollar allocation, no need to scale separately

        # Currency
        currency = "USD"
        if "Crncy" in df.columns and not pd.isna(row.get("Crncy")):
            currency = str(row["Crncy"]).strip()

        # FX
        fx_rate = 1.0
        if "FX Cls" in df.columns and not pd.isna(row.get("FX Cls")) and row["FX Cls"] != 0:
            fx_rate = row["FX Cls"]

        px_close = row.get("Px Close", 0.0)
        if pd.isna(px_close) or px_close <= 0:
            continue

        price_usd = px_close * fx_rate
        
        # Distinguish options if ticker looks like FXI 6 P30, etc.
        is_option = bool(option_pattern.search(ticker_raw))

        # Calculate theoretical quantity (fractional)
        theo_qty = target_usd / price_usd
        
        # Skip positions that would be too small
        if abs(target_usd) < min_allocation:
            continue
            
        theoretical_positions.append({
            "Ticker": ticker_raw,
            "Is Option?": is_option,
            "Weight": w_float,
            "Currency": currency,
            "FX Rate": fx_rate,
            "Price USD": price_usd,
            "Theoretical Qty": theo_qty,
            "Target USD": target_usd,
            "Direction": 1 if target_usd >= 0 else -1
        })
    
    # Convert to DataFrame for easier manipulation
    theo_df = pd.DataFrame(theoretical_positions)
    
    if theo_df.empty:
        print("[WARNING] No valid positions found after filtering.")
        orders_df = pd.DataFrame([])
        return orders_df
    
    # Calculate the total theoretical allocation (with fractional shares)
    theo_df["Final USD"] = theo_df["Theoretical Qty"] * theo_df["Price USD"]
    theo_df["Final Qty"] = theo_df["Theoretical Qty"]
    total_theoretical = theo_df["Final USD"].sum()
    
    print(f"[DEBUG] Theoretical allocation (with fractional shares): ${total_theoretical:.2f}")
    print(f"[DEBUG] Target: ${net_investment:.2f}")
    print(f"[DEBUG] Difference: ${total_theoretical - net_investment:.2f} ({(total_theoretical/net_investment - 1)*100:.4f}%)")
    
    # Now round the quantities and calculate the impact of rounding
    theo_df["Rounded Qty"] = theo_df.apply(
        lambda x: math.ceil(abs(x["Final Qty"])) if round_up else math.floor(abs(x["Final Qty"])),
        axis=1
    )
    
    # Filter out zero quantities
    theo_df, zero_df = theo_df[theo_df["Rounded Qty"] > 0], theo_df[theo_df["Rounded Qty"] == 0]
    
    if theo_df.empty:
        print("[WARNING] No positions with non-zero quantities after rounding.")
        orders_df = pd.DataFrame([])
        return orders_df
    else:
        # These positions have zero quantities
        if not zero_df.empty:
            print(f"[DEBUG] Found {len(zero_df)} positions with zero quantities after rounding.")
        else:
            print("[DEBUG] No positions with zero quantities after rounding.")
    
    # Calculate actual USD values after rounding
    theo_df["Actual USD"] = theo_df["Rounded Qty"] * theo_df["Price USD"]
    theo_df["Absolute USD Exposure"] = theo_df["Actual USD"] * theo_df["Direction"]
    
    # Calculate the rounding impact for each position
    theo_df["Rounding Impact"] = theo_df["Actual USD"] - abs(theo_df["Final USD"])
    
    # Calculate total allocation after rounding
    total_long = theo_df[theo_df["Direction"] > 0]["Actual USD"].sum()
    total_short = abs(theo_df[theo_df["Direction"] < 0]["Absolute USD Exposure"].sum())
    total_net = total_long - total_short
    
    print(f"[DEBUG] Initial allocation after rounding:")
    print(f"[DEBUG] Long: ${total_long:.2f}, Short: ${total_short:.2f}, Net: ${total_net:.2f}")
    print(f"[DEBUG] Target: ${net_investment:.2f}")
    print(f"[DEBUG] Difference: ${total_net - net_investment:.2f} ({(total_net/net_investment - 1)*100:.2f}%)")

    final_positions = theo_df.copy()
    
    # Calculate final total
    final_total = final_positions["Actual USD"].sum()
    final_diff = abs(total_net - net_investment)
    print(f"[DEBUG] After adjustment: ${final_total:.2f} (diff: ${final_diff:.2f}, {final_diff/net_investment*100:.2f}%)")
    
    # Create final orders
    orders = []
    
    for _, row in final_positions.iterrows():
        action = "SELL SHORT" if row["Direction"] < 0 else "BUY"
        
        orders.append({
            "Ticker": row["Ticker"],
            "Is Option?": row["Is Option?"],
            "Weight Col": weight_col,
            "Raw Weight Used": row["Weight"] * 100,  # Convert back to percentage for display
            "Converted Weight": row["Weight"],
            "Currency": row["Currency"],
            "FX Rate": row["FX Rate"],
            "Local Px": row["Price USD"] / row["FX Rate"],
            "USD Px": row["Price USD"],
            "Theoretical Qty": abs(row["Theoretical Qty"]),  # Add theoretical quantity for reference
            "Quantity": row["Rounded Qty"],
            "Target USD Value": row["Target USD"],
            "Theoretical USD Value": row["Target USD"],  # Changed to show the target instead of the incorrectly signed "Final USD"
            "Actual USD Value": -row["Actual USD"] if row["Direction"] < 0 else row["Actual USD"],  # Make the value negative for short positions
            "Rounding Impact": row["Actual USD"] - abs(row["Final USD"]),  # Recalculate rounding impact
            "Action": action,
            "Net Order Qty": row["Rounded Qty"]  # Default, will be updated if current_portfolio is provided
        })
    
    orders_df = pd.DataFrame(orders)
    
    # Final allocation stats
    total_long = orders_df[orders_df["Action"] == "BUY"]["Actual USD Value"].sum()
    total_short = abs(orders_df[orders_df["Action"] == "SELL SHORT"]["Actual USD Value"].sum())
    total_net = total_long - total_short
    
    print(f"[DEBUG] Created {len(orders_df)} order rows")
    print(f"[DEBUG] Final allocation:")
    print(f"[DEBUG] Long: ${total_long:.2f}")
    print(f"[DEBUG] Short: ${total_short:.2f}")
    print(f"[DEBUG] Net: ${total_net:.2f}")
    print(f"[DEBUG] Target: ${net_investment:.2f}")
    print(f"[DEBUG] Difference: ${total_net - net_investment:.2f} ({(total_net/net_investment - 1)*100:.2f}%)")

    if output_file:
        # Write to Excel
        with pd.ExcelWriter(output_file) as writer:
            orders_df.to_excel(writer, sheet_name="Parsed Orders", index=False)
        
        # Only split into stocks/options if we have data
        if not orders_df.empty:
            # Also can split out options vs. stocks
            stocks_df = orders_df[~orders_df["Is Option?"]]
            opts_df = orders_df[orders_df["Is Option?"]]
            
            # Create 'orders' directory if it doesn't exist
            os.makedirs("orders", exist_ok=True)
            
            # Save different versions of the orders
            stocks_df.to_csv("orders/stocks_orders.csv", index=False)
            opts_df.to_csv("orders/options_orders.csv", index=False)
            
            # Create a clean version with only net orders (skipping zero quantity orders)
            net_orders = orders_df[orders_df["Net Order Qty"] != 0].copy()
            net_orders["Action"] = net_orders.apply(
                lambda x: "BUY" if x["Net Order Qty"] > 0 and x["Action"] != "SELL SHORT" 
                     else "SELL SHORT" if x["Action"] == "SELL SHORT" 
                     else "SELL" if x["Net Order Qty"] < 0 
                     else "NONE",
                axis=1
            )
            net_orders.to_csv("orders/net_orders.csv", index=False)
            
            # Also create a clean Schwab-format CSV with just the essentials
            schwab_format = net_orders[["Ticker", "Action", "Net Order Qty", "USD Px"]].copy()
            schwab_format.columns = ["Symbol", "Action", "Quantity", "Price"]
            schwab_format["Action"] = schwab_format["Action"].map({"BUY": "Buy", "SELL": "Sell", "SELL SHORT": "Sell Short"})
            schwab_format["Quantity"] = schwab_format["Quantity"].abs()
            schwab_format.to_csv("orders/schwab_upload.csv", index=False)
            
            print("[DEBUG] Wrote final order files to 'orders/' directory")
        else:
            print("[WARNING] No orders were generated. Check your input file and parameters.")

    return orders_df

def fetch_current_price(ticker):
    """
    Fetch the current market price for a ticker using yfinance.
    Includes error handling and rate limiting.
    
    Parameters
    ----------
    ticker: str
        The ticker symbol to look up
        
    Returns
    -------
    tuple
        (price, currency) or (None, None) if not found
    """
    # Remove any exchange suffix (e.g., " US")
    clean_ticker = ticker.split()[0] if " " in ticker else ticker
    
    try:
        # Add rate limiting to avoid API throttling
        time.sleep(0.5)
        
        # Fetch data
        ticker_data = yf.Ticker(clean_ticker)
        info = ticker_data.info
        
        if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            price = info['regularMarketPrice']
            currency = info.get('currency', 'USD')
            return price, currency
        else:
            print(f"[DEBUG] Could not find price for {ticker}")
            return None, None
            
    except Exception as e:
        print(f"[DEBUG] Error fetching price for {ticker}: {str(e)}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Citrindex portfolio and generate Schwab orders')
    parser.add_argument('input_file', help='Path to Citrindex portfolio Excel file')
    parser.add_argument('--net-investment', type=float, default=1000000, 
                        help='Total capital to allocate (long - short)')
    parser.add_argument('--output-file', default="schwab_orders.xlsx",
                        help='Output Excel file for orders')
    parser.add_argument('--min-allocation', type=float, default=0,
                        help='Minimum allocation in USD')
    parser.add_argument('--round-up', action='store_true', default=True,
                        help='Round up share quantities (default: True)')
    parser.add_argument('--use-delta', action='store_true', default=False,
                        help='Use Delta Adj Wgt if available (default: False)')
    parser.add_argument('--us-only', action='store_true', default=True,
                        help='Filter out non-US stocks (default: True)')
    parser.add_argument('--no-us-only', action='store_false', dest='us_only',
                        help='Include non-US stocks')
    parser.add_argument('--find-us-replacements', action='store_true', default=False,
                        help='Try to find US replacements for non-US stocks (default: False)')
    parser.add_argument('--update-prices', action='store_true', default=False,
                        help='Fetch current market prices for tickers (especially for US replacements)')
    parser.add_argument('--current-portfolio', help='Path to Excel file containing current Schwab portfolio for calculating net orders')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        exit(1)
    
    try:
        # Process the file
        process_combined_sheet(
            args.input_file,
            net_investment=args.net_investment,
            output_file=args.output_file,
           #exclude=["CASH", "CASH (USD)", "CASH (USD) (USD)", "CASH (USD) (USD) (USD)"],
            use_delta=args.use_delta,
            min_allocation=args.min_allocation,
            round_up=args.round_up,
            us_only=args.us_only,
            find_us_replacements=args.find_us_replacements,
            update_prices=args.update_prices,
            current_portfolio=args.current_portfolio
        )
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        exit(1)