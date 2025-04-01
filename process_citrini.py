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

# --- Helper Function to Normalize Tickers ---
def normalize_ticker(ticker):
    """
    Normalizes ticker symbols for better matching between sources.
    Removes exchange suffixes, handles options, etc.
    """
    if not isinstance(ticker, str):
        return None
    
    ticker = ticker.upper().strip()
    
    # Basic option check (simple pattern, might need refinement)
    # Example: "LLY 06/20/2025 1000.00 C" -> "LLY"
    option_match = re.match(r"^([A-Z./]+)\\s+\\d{2}/\\d{2}/\\d{4}\\s+[\\d.]+\\s+[CP]$", ticker)
    if option_match:
        return option_match.group(1)
        
    # Remove common exchange suffixes like ' US', '.US', ' NA' etc.
    ticker = re.sub(r'\\s+(US|NA|LN|PA|HK|JP|SW|DE|CA)$', '', ticker)
    ticker = re.sub(r'\\.(US|NA|L|PA|HK|T|SW|DE|TO)$', '', ticker) # e.g. BRK.B
        
    # Handle cases like "BF/B" -> "BF.B" (common in yfinance)
    ticker = ticker.replace('/', '.')

    # Extract base ticker if it contains spaces (like ADR descriptions)
    # Example: "ASML HLDG N V FSPONSORED ADR..." -> "ASML"
    # Only do this if it doesn't look like an option string we missed
    if ' ' in ticker and not re.search(r'\\d{2}/\\d{2}/\\d{4}', ticker):
         parts = ticker.split()
         # Assume the first part consisting only of letters/dots is the ticker
         if re.match(r'^[A-Z.]+$', parts[0]):
             ticker = parts[0]

    return ticker


# --- Function to Read Schwab Portfolio ---
def read_schwab_portfolio(csv_path):
    """
    Reads and processes the Schwab positions CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the Schwab Individual-Positions.csv file.

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing current holdings (Symbol, Current Qty, Security Type)
        or None if the file cannot be processed.
    """
    try:
        # Read the CSV, skipping initial rows and using the correct header
        df = pd.read_csv(csv_path, header=2)
        print(f"[DEBUG] Read Schwab portfolio: {len(df)} rows initially.")
        
        # Check for essential columns
        required_cols = ["Symbol", "Qty (Quantity)", "Security Type", "Price"]
        if not all(col in df.columns for col in required_cols):
            print(f"[ERROR] Schwab CSV missing required columns. Found: {df.columns.tolist()}")
            return None
            
        # Filter out summary rows at the end (like 'Account Total', 'Cash & Cash Investments')
        df = df[~df['Symbol'].isin(['Account Total', 'Cash & Cash Investments', '--'])]
        df = df.dropna(subset=['Symbol', 'Qty (Quantity)'])
        print(f"[DEBUG] Schwab portfolio after dropping summary/NA rows: {len(df)} rows.")

        # Select and rename columns
        df = df[["Symbol", "Qty (Quantity)", "Security Type", "Price"]].copy()
        df.rename(columns={
            "Qty (Quantity)": "Current Qty",
            "Symbol": "Schwab Symbol",
            "Price": "Schwab Price"
            }, inplace=True)

        # Clean Quantity: remove commas, convert to numeric
        df["Current Qty"] = df["Current Qty"].astype(str).str.replace(',', '', regex=False)
        df["Current Qty"] = pd.to_numeric(df["Current Qty"], errors='coerce')
        
        # Clean Price: remove '$', convert to numeric
        df["Schwab Price"] = df["Schwab Price"].astype(str).str.replace('$', '', regex=False)
        df["Schwab Price"] = pd.to_numeric(df["Schwab Price"], errors='coerce')

        # Normalize the Schwab Symbol
        df['Symbol'] = df['Schwab Symbol'].apply(normalize_ticker)
        
        # Filter out rows where quantity is NaN or Symbol normalization failed
        df = df.dropna(subset=['Current Qty', 'Symbol'])
        print(f"[DEBUG] Schwab portfolio after cleaning Qty/Symbol: {len(df)} rows.")

        # Keep only relevant security types (adjust as needed)
        # relevant_types = ["Equity", "ETFs & Closed End Funds", "Option"]
        # df = df[df['Security Type'].isin(relevant_types)]
        # print(f"[DEBUG] Schwab portfolio after filtering for relevant types: {len(df)} rows.")
        # Decided to keep all types for now to allow matching everything,
        # the target portfolio filtering will handle relevance.

        # Set Symbol as index for easy lookup
        df = df.set_index('Symbol')
        
        print(f"[DEBUG] Processed Schwab portfolio. Found {len(df)} positions.")
        print("[DEBUG] Sample current portfolio data:")
        print(df.head())
        
        return df[['Current Qty', 'Schwab Price', 'Schwab Symbol', 'Security Type']] # Return relevant columns

    except FileNotFoundError:
        print(f"[ERROR] Schwab portfolio file not found: {csv_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to read or process Schwab portfolio '{csv_path}': {e}")
        return None


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
    # Fallback: normalize the original ticker itself
    return normalize_ticker(ticker)


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
    # Normalize ticker first
    clean_ticker = normalize_ticker(ticker)
    if not clean_ticker:
        print(f"[DEBUG] Invalid ticker for price fetch: {ticker}")
        return None, None
        
    try:
        # Add rate limiting to avoid API throttling
        time.sleep(0.1) # Reduce sleep time slightly
        
        # Fetch data
        ticker_data = yf.Ticker(clean_ticker)
        info = ticker_data.info

        price = None
        # Try different price fields
        if 'currentPrice' in info and info['currentPrice'] is not None:
             price = info['currentPrice']
        elif 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            price = info['regularMarketPrice']
        elif 'ask' in info and info['ask'] is not None and info['ask'] > 0:
             price = info['ask'] # Use ask if available and non-zero
        elif 'bid' in info and info['bid'] is not None and info['bid'] > 0:
            price = info['bid'] # Use bid if ask is not available
        elif 'previousClose' in info and info['previousClose'] is not None:
            price = info['previousClose'] # Fallback to previous close

        if price is not None:
            currency = info.get('currency', 'USD')
            print(f"[DEBUG] Fetched price for {clean_ticker}: {price} {currency}")
            return price, currency
        else:
            # Attempt to get history if info failed
            hist = ticker_data.history(period="1d")
            if not hist.empty:
                 last_close = hist['Close'].iloc[-1]
                 currency = info.get('currency', 'USD') # Still try to get currency
                 print(f"[DEBUG] Fetched price for {clean_ticker} via history: {last_close} {currency}")
                 return last_close, currency
            else:
                 print(f"[DEBUG] Could not find price for {clean_ticker} via info or history.")
                 return None, None
            
    except Exception as e:
        print(f"[DEBUG] Error fetching price for {clean_ticker}: {str(e)}")
        return None, None

def process_combined_sheet(
    input_file,
    target_gross_exposure,
    output_file=None,
    exclude=None,
    use_delta=False,
    min_allocation=0,
    round_up=True,
    us_only=False,
    find_us_replacements=False,
    update_prices=False,
    current_portfolio_csv=None
):
    """
    Reads the single Excel sheet, compares with current portfolio,
    and creates net orders.

    (Parameters description mostly the same, added current_portfolio_csv)
    ... [rest of docstring] ...
    current_portfolio_csv: str or None
        Path to Schwab CSV file containing current portfolio holdings.
    """
    if exclude is None:
        exclude = []

    # --- 1) Load Citrindex Data ---
    xls = pd.ExcelFile(input_file)
    df = None
    chosen_sheet = None
    for sheet in xls.sheet_names:
        try:
             # Try reading with different header rows
             for header_row in range(0, 20):
                 temp_df = pd.read_excel(input_file, sheet_name=sheet, header=header_row)
                 needed = ["Ticker", "Px Close"] # Minimal check
                 if all(col in temp_df.columns for col in needed):
                      # More robust check for non-empty Ticker data
                      if not temp_df["Ticker"].dropna().empty:
                           df = temp_df
                           chosen_sheet = sheet
                           print(f"[DEBUG] Found target data in sheet '{chosen_sheet}' with header row {header_row}")
                           break
                 if df is not None: break
             if df is not None: break
        except Exception as e:
             print(f"[WARNING] Could not read sheet '{sheet}' properly: {e}")
             continue


    if df is None:
        raise ValueError("No sheet found matching the required columns (Ticker, Px Close, etc.) with valid data.")

    print(f"[DEBUG] Columns found: {df.columns.tolist()}")
    print(f"[DEBUG] Sample target data (first 3 rows):")
    pd.set_option('display.max_columns', None) # Ensure all columns are shown
    print(df.head(3))
    print(f"[DEBUG] Total rows in target input: {len(df)}")

    df = df.dropna(subset=["Ticker"])
    df = df[df["Ticker"].astype(str).str.strip() != ""]
    print(f"[DEBUG] Target after removing blank tickers: {len(df)} rows")

    # Exclude summary rows (improved logic)
    summary_keywords = [
        "CITRINDEX:", "AIRLINES BASKET", "CHINESE EQUITY BARBELL", "CITRINI SMID CAP PICKS",
        "FISCAL PRIMACY", "GLOBAL AI BASKET", "MEDTECH & HEALTHCARE", "NAT GAS", "RTHEDG",
        "TOTAL", "SUMMARY", "FOOTNOTE", "GRAND", "CASH", "AGGREGATE", "NET EXPOSURE",
        "GROSS EXPOSURE", "LONG EXPOSURE", "SHORT EXPOSURE", "PORTFOLIO STATISTICS",
        "SUBTOTAL", "BENCHMARK", "SECTOR" # Added more keywords
    ]
    
    # Check multiple potential columns for summary keywords
    potential_summary_cols = [col for col in df.columns if df[col].dtype == 'object'][:5] # Check first 5 object columns
    potential_summary_cols.append("Ticker") # Always check Ticker
    
    mask = pd.Series(True, index=df.index)
    for col in potential_summary_cols:
         if col in df.columns:
              # Case-insensitive check, handle non-string data
              mask &= ~df[col].astype(str).str.upper().str.contains('|'.join(summary_keywords), na=False)

    df = df[mask]
    print(f"[DEBUG] Target after excluding summary rows: {len(df)} rows")

    # *** ADD: Preserve original order ***
    df = df.reset_index(drop=True).reset_index().rename(columns={'index': 'original_order'})

    # Normalize Ticker early
    df['Original Ticker'] = df['Ticker']
    df['Ticker'] = df['Ticker'].apply(normalize_ticker)
    # Handle potential NaN tickers resulting from normalization
    df = df.dropna(subset=['Ticker'])
    df = df[df['Ticker'] != ''] # Also remove empty strings if any
    print(f"[DEBUG] Target after normalizing tickers: {len(df)} rows")

    # Exclude user-specified tickers (apply to normalized ticker)
    if exclude: # Check if exclude list is not empty
        normalized_exclude = [normalize_ticker(t) for t in exclude if normalize_ticker(t)]
        # Add check to ensure normalized_exclude doesn't contain None or empty strings
        normalized_exclude = [t for t in normalized_exclude if t]
        if normalized_exclude: # Check if there are any valid tickers to exclude
             df = df[~df["Ticker"].isin(normalized_exclude)]
             print(f"[DEBUG] Target after excluding user-specified tickers: {len(df)} rows")
        else:
             print("[DEBUG] No valid tickers provided in the exclude list.")
    else:
        print("[DEBUG] No user-specified tickers to exclude.")

    # --- Handle US vs Non-US ---
    # Convert currency to uppercase and handle NaNs
    df['Crncy'] = df['Crncy'].fillna('USD').astype(str).str.upper()
    
    non_us_mask = ~(df["Ticker"].str.contains(" ") & df["Ticker"].str.endswith(" US")) & (df["Crncy"] != "USD")
    
    if find_us_replacements:
        non_us_tickers = df.loc[non_us_mask, "Original Ticker"].unique().tolist()
        print(f"[DEBUG] Attempting to find US replacements for {len(non_us_tickers)} potential non-US tickers.")
        replacements_found = 0
        for i, original_ticker in enumerate(non_us_tickers):
            if i > 0 and i % 5 == 0:
                print(f"[DEBUG] Pausing for rate limiting during US ticker lookup ({i}/{len(non_us_tickers)})...")
                time.sleep(1)
            
            normalized_original = normalize_ticker(original_ticker)
            print(f"[DEBUG] Looking up replacement for: {original_ticker} (normalized: {normalized_original})")
            us_ticker_replacement = lookup_us_ticker(original_ticker) # Use original for lookup
            
            # Check if a meaningful replacement was found and is different
            if us_ticker_replacement and us_ticker_replacement != normalized_original and not any(kw in us_ticker_replacement for kw in summary_keywords):
                print(f"[DEBUG] Found US replacement: {original_ticker} -> {us_ticker_replacement}")
                # Update the 'Ticker' column for rows matching the *original* normalized ticker
                df.loc[df['Ticker'] == normalized_original, 'Ticker'] = us_ticker_replacement
                df.loc[df['Ticker'] == us_ticker_replacement, 'Crncy'] = 'USD'
                df.loc[df['Ticker'] == us_ticker_replacement, 'FX Cls'] = 1.0
                
                 # Optionally update the price for the new US ticker
                if update_prices:
                     current_price, currency = fetch_current_price(us_ticker_replacement)
                     if current_price is not None:
                         print(f"[DEBUG] Updated price for replacement {us_ticker_replacement}: ${current_price:.2f}")
                         df.loc[df['Ticker'] == us_ticker_replacement, 'Px Close'] = current_price
                     else:
                          print(f"[WARNING] Could not fetch price for replacement {us_ticker_replacement}. Keeping original price.")
                replacements_found += 1
            else:
                 print(f"[DEBUG] No distinct US replacement found for {original_ticker}.")

        print(f"[DEBUG] Found US replacements for {replacements_found} non-US tickers.")
        # Re-evaluate non_us_mask after replacements
        non_us_mask = ~(df["Ticker"].str.contains(" ") & df["Ticker"].str.endswith(" US")) & (df["Crncy"] != "USD")


    if us_only:
        non_us_count = non_us_mask.sum()
        if non_us_count > 0:
            print(f"[DEBUG] Filtering out {non_us_count} non-US stocks because us_only=True.")
            print("[DEBUG] Non-US tickers being removed:")
            print(df.loc[non_us_mask, 'Original Ticker'].tolist())
            df = df[~non_us_mask]
            print(f"[DEBUG] Target after filtering for US stocks only: {len(df)} rows")
        else:
             print("[DEBUG] No non-US stocks to filter out (or replacements were found).")


    # --- 2) Clean Target Data ---
    # Convert price and FX columns to numeric, handle errors
    df['Px Close'] = pd.to_numeric(df['Px Close'], errors='coerce')
    df['FX Cls'] = pd.to_numeric(df.get('FX Cls', 1.0), errors='coerce').fillna(1.0) # Default FX to 1.0 if missing/invalid
    df['Price USD'] = df['Px Close'] * df['FX Cls']
    
    # Drop rows with invalid prices
    df = df.dropna(subset=['Price USD'])
    df = df[df['Price USD'] > 0]
    print(f"[DEBUG] Target after cleaning prices: {len(df)} rows")

    # --- 3) Determine Weights ---
    if use_delta and "Delta Adj Wgt" in df.columns:
        weight_col = "Delta Adj Wgt"
        print("[DEBUG] Using 'Delta Adj Wgt' as the weighting column.")
    elif "Wgt" in df.columns:
        weight_col = "Wgt"
        print("[DEBUG] Using 'Wgt' as the weighting column.")
    elif "Market Value (%)" in df.columns:
         weight_col = "Market Value (%)"
         print("[DEBUG] Using 'Market Value (%)' as the weighting column.")
    else:
        # Handle case where no weight column found even before aggregation
        raise ValueError("Could not find a suitable weighting column ('Delta Adj Wgt', 'Wgt', or 'Market Value (%)').")

    # Convert chosen weight column to numeric, coercing errors
    df['Raw Weight'] = pd.to_numeric(df[weight_col], errors='coerce')
    df = df.dropna(subset=['Raw Weight'])
    print(f"[DEBUG] Target after cleaning weights: {len(df)} rows")
    
    # Filter extreme weights
    df = df[df['Raw Weight'].abs() < 500] # Filter weights > 500%
    print(f"[DEBUG] Target after filtering extreme weights: {len(df)} rows")


    # --- Aggregate Targets Before Normalization --- 
    aggregation_functions = {
        'Raw Weight': 'sum',
        'Original Ticker': 'first',
        'original_order': 'first', # *** ADD: Keep original order index ***
        # Make sure these columns actually exist in df before trying to aggregate them
        'Crncy': 'first' if 'Crncy' in df.columns else None,
        'FX Cls': 'first' if 'FX Cls' in df.columns else None,
        'Px Close': 'first' if 'Px Close' in df.columns else None,
        'Price USD': 'first' if 'Price USD' in df.columns else None,
        # Add the actual weight column used to the aggregation
        weight_col: 'first' # Keep the first instance of the original weight value for reference
    }
    # Filter out None values from functions dict and assign to cols_to_agg
    cols_to_agg = {k: v for k, v in aggregation_functions.items() if v is not None and k in df.columns} # Check column exists

    # Check if Raw Weight exists (should always after previous step)
    if 'Raw Weight' not in df.columns:
         raise ValueError("Required 'Raw Weight' column missing for aggregation.")

    # Store index name if it exists
    index_name = df.index.name

    # Perform aggregation
    print(f"[DEBUG] Aggregating target data by normalized ticker. Initial rows: {len(df)}")
    # Ensure the group keys don't become the index immediately if not desired
    df_agg = df.groupby('Ticker', as_index=False).agg(cols_to_agg).reset_index(drop=True)

    print(f"[DEBUG] Target after aggregation by ticker: {len(df_agg)} rows")
    
    # Check for duplicates after aggregation (shouldn't be any by 'Ticker')
    if df_agg['Ticker'].duplicated().any():
        print("[WARNING] Duplicate tickers found even after aggregation. Check aggregation logic.")
        print(df_agg[df_agg['Ticker'].duplicated()]['Ticker'])

    # Use the aggregated DataFrame from now on
    df = df_agg

    # --- Normalize Aggregated Weights ---
    total_abs_weight = df["Raw Weight"].abs().sum()
    if total_abs_weight == 0:
        print("[WARNING] Total absolute weight is zero after aggregation. Cannot calculate allocations.")
        return pd.DataFrame()
        
    df["NormWeight"] = df["Raw Weight"] / total_abs_weight
    total_long_norm = df.loc[df["NormWeight"] > 0, "NormWeight"].sum()
    total_short_norm = abs(df.loc[df["NormWeight"] < 0, "NormWeight"].sum())
    print(f"[DEBUG] Normalized Long Weight Sum: {total_long_norm:.4f}")
    print(f"[DEBUG] Normalized Short Weight Sum: {total_short_norm:.4f}")
    print(f"[DEBUG] Check sum AFTER normalization: {df['NormWeight'].sum():.4f} (should be close to Long - Short)")

    # --- Calculate Target Allocation based on Gross Exposure ---
    df["Target USD"] = df["NormWeight"] * target_gross_exposure
    allocated_long = df.loc[df["NormWeight"] > 0, "Target USD"].sum()
    allocated_short = abs(df.loc[df["NormWeight"] < 0, "Target USD"].sum())
    calculated_gross = allocated_long + allocated_short
    calculated_net = allocated_long - allocated_short
    print(f"[DEBUG] Target Allocated to longs : ${allocated_long:,.2f}")
    print(f"[DEBUG] Target Allocated to shorts: ${allocated_short:,.2f}")
    print(f"[DEBUG] Implied Target Gross: ${calculated_gross:,.2f} (Target Input: ${target_gross_exposure:,.2f})")
    print(f"[DEBUG] Implied Target Net  : ${calculated_net:,.2f}")


    # --- Update Prices (for the aggregated df) ---
    if update_prices:
        print("[DEBUG] Updating prices for aggregated target tickers...")
        prices_updated = 0
        tickers_to_update = df["Ticker"].unique().tolist() # Tickers are now unique
        for i, ticker in enumerate(tickers_to_update):
            if i > 0 and i % 10 == 0:
                print(f"[DEBUG] Pausing for rate limiting during price update ({i}/{len(tickers_to_update)})...")
                time.sleep(1)
            
            current_price, currency = fetch_current_price(ticker)
            if current_price is not None:
                 # Update the Price USD directly
                 df.loc[df['Ticker'] == ticker, 'Price USD'] = current_price
                 # Also update Px Close and FX Cls based on currency
                 df.loc[df['Ticker'] == ticker, 'Px Close'] = current_price
                 df.loc[df['Ticker'] == ticker, 'FX Cls'] = 1.0 if currency == 'USD' else df.loc[df['Ticker'] == ticker, 'FX Cls'] # Keep original FX if not USD
                 df.loc[df['Ticker'] == ticker, 'Currency'] = currency # Update currency
                 prices_updated += 1
        print(f"[DEBUG] Updated prices for {prices_updated} aggregated target tickers.")
        # Recalculate Price USD if prices were updated
        # Ensure Px Close and FX Cls are numeric before multiplying
        df['Px Close'] = pd.to_numeric(df['Px Close'], errors='coerce')
        df['FX Cls'] = pd.to_numeric(df['FX Cls'], errors='coerce')
        df['Price USD'] = (df['Px Close'] * df['FX Cls']).fillna(df['Price USD']) # Keep old Price USD if calc fails


    # --- Calculate Target Quantities (Theoretical & Rounded) ---
    # Ensure Price USD exists and is valid before division
    df = df.dropna(subset=['Price USD'])
    df = df[df['Price USD'] > 0]
    print(f"[DEBUG] Aggregated target before Qty calc: {len(df)} rows")
    
    df['Theoretical Qty'] = df['Target USD'] / df['Price USD']
    df = df[df['Theoretical Qty'].notna()] # Remove rows where qty calculation failed (e.g., Price USD was zero)
    
    df = df[df['Target USD'].abs() >= min_allocation]
    print(f"[DEBUG] Aggregated target after minimum allocation filter: {len(df)} rows")

    round_func = np.ceil if round_up else np.floor
    # RENAME Target Qty to Ideal Target Qty for clarity in this step
    df['Ideal Target Qty'] = (round_func(df['Theoretical Qty'].abs()) * np.sign(df['Theoretical Qty'])).fillna(0).astype(int)

    # Filter out positions that round to zero quantity IN THE TARGET
    df = df[df['Ideal Target Qty'] != 0] 
    print(f"[DEBUG] Aggregated target after rounding and removing zero quantities: {len(df)} rows")
    
    if df.empty:
         print("[WARNING] No target positions remaining after aggregation and filtering.")
         # Still need to potentially process exit orders if current portfolio exists
         # So, create an empty target_df but continue if current_portfolio_csv is provided
         target_df = pd.DataFrame(columns=['original_order', 'Normalized Ticker', 'Original Ticker', 'Ideal Target Qty', 'Price USD', 'Target USD', 
                                          'Raw Weight', 'NormWeight', 'Crncy', 'FX Cls', 'Px Close', weight_col])
         target_df = target_df.set_index('Normalized Ticker')
         if not current_portfolio_csv:
              return pd.DataFrame() # Exit if no target AND no current portfolio
    else:
        # Prepare final target DataFrame for merging (now unique by Ticker)
        # Select columns that definitely exist after aggregation
        target_cols_to_keep = ['original_order', 'Ticker', 'Original Ticker', 'Ideal Target Qty', 'Price USD', 'Target USD', # Added original_order
                               'Raw Weight', 'NormWeight', 'Crncy', 'FX Cls', 'Px Close', weight_col]
        target_cols_present = [col for col in target_cols_to_keep if col in df.columns]
        # Make sure 'Ticker' is definitely present for index setting
        if 'Ticker' not in target_cols_present: target_cols_present.insert(0, 'Ticker')
        target_df = df[target_cols_present].copy()
        target_df = target_df.rename(columns={'Ticker': 'Normalized Ticker'})
        # Set index, DROP the column this time
        target_df = target_df.set_index('Normalized Ticker', drop=True)


    # --- Load Current Portfolio --- 
    current_portfolio_df = None
    if current_portfolio_csv:
        print(f"\n[INFO] Loading current portfolio from: {current_portfolio_csv}")
        current_portfolio_df = read_schwab_portfolio(current_portfolio_csv)
        if current_portfolio_df is None:
             print("[WARNING] Could not load current portfolio. Calculating total orders instead of net orders.")
             # If target is also empty, exit
             if target_df.empty:
                  return pd.DataFrame()
    # If no current portfolio CSV, and target was empty, we already returned
    elif target_df.empty:
         print("[ERROR] No target data and no current portfolio data provided.")
         return pd.DataFrame()
    else:
         print("\n[INFO] No current portfolio provided. Calculating total target orders.")
         # Create an empty current_portfolio_df with the expected index name
         current_portfolio_df = pd.DataFrame(columns=['Current Qty', 'Schwab Price', 'Schwab Symbol', 'Security Type'])
         current_portfolio_df.index.name = 'Normalized Ticker'


    # --- Merge Target and Current Portfolios ---
    print("[DEBUG] Merging TARGET portfolio with current holdings...")
    # Join on the index (Normalized Ticker for both)
    # Outer join remains correct to capture all tickers from both sources
    merged_df = target_df.join(current_portfolio_df, how='outer', lsuffix='_target') # index is Normalized Ticker

    # Reset index to make Normalized Ticker a regular column.
    merged_df.reset_index(inplace=True)

    # Verify the column name after reset_index (usually the index name, or 'index')
    if 'Normalized Ticker' not in merged_df.columns:
        if 'index' in merged_df.columns:
            merged_df.rename(columns={'index': 'Normalized Ticker'}, inplace=True)
        else:
            raise ValueError("Normalized Ticker column not found after reset_index.")

    # --- Fill NaNs and Prepare for Net Order Calculation ---
    # Fill NaNs introduced by the outer join
    # For target columns, NaN means the stock is only in current portfolio (target is 0)
    merged_df['Ideal Target Qty'] = merged_df['Ideal Target Qty'].fillna(0).astype(int)
    merged_df['Original Target Ticker'] = merged_df.get('Original Ticker', pd.Series(index=merged_df.index)).fillna('')
    # Use Schwab price if target price missing (for exit orders) 
    if 'USD Price Used' not in merged_df.columns:
        # Use Price USD from target if available, else create NaN column
        merged_df['USD Price Used'] = merged_df.get('Price USD', np.nan)
    if 'Schwab Price' in merged_df.columns:
        merged_df['USD Price Used'] = merged_df['USD Price Used'].fillna(merged_df['Schwab Price'])
    merged_df['Target USD'] = merged_df['Target USD'].fillna(0)
    # Fill NaN weights/prices from target side with 0 or NaN
    for col in ['Raw Weight', 'NormWeight', 'Px Close', 'FX Cls', weight_col]: # Include original weight col
         if col in merged_df.columns:
              merged_df[col] = merged_df[col].fillna(0 if 'Weight' in col else np.nan)
         # Ensure column exists even if not in target_df (will be all NaN/0)
         elif col not in merged_df.columns:
              merged_df[col] = 0 if 'Weight' in col else np.nan 

    # For current columns, NaN means the stock is only in target portfolio (current is 0)
    merged_df['Current Qty'] = merged_df['Current Qty'].fillna(0).astype(int)
    if 'Schwab Symbol' not in merged_df.columns:
         merged_df['Schwab Symbol'] = merged_df['Normalized Ticker']
    else:
         merged_df['Schwab Symbol'] = merged_df['Schwab Symbol'].fillna(merged_df['Normalized Ticker'])
    merged_df['Security Type'] = merged_df.get('Security Type', pd.Series(index=merged_df.index)).fillna('N/A')

    # *** ADD: Fill original_order NaN with a large number for sorting last ***
    merged_df['original_order'] = merged_df['original_order'].fillna(np.inf)

    # Drop rows where we couldn't determine a price (essential for calculation)
    merged_df = merged_df.dropna(subset=['USD Price Used'])
    print(f"[DEBUG] Merged DataFrame (outer join) after NaN handling: {len(merged_df)} rows.")


    # --- Calculate Net Orders (Target - Current) ---
    # Use 'Ideal Target Qty' here
    merged_df['Net Order Qty'] = merged_df['Ideal Target Qty'] - merged_df['Current Qty'] 

    # Determine Action based on Net Order Quantity and Current Qty
    def determine_action(row):
        net_qty = row['Net Order Qty']
        current_qty = row['Current Qty']
        # Use Ideal Target Qty for logic consistency
        target_qty = row['Ideal Target Qty'] 

        if net_qty > 0:
            return "BUY" # Buying more or initiating long
        elif net_qty < 0:
            # Selling shares we own or Buying to cover short
            if current_qty > 0:
                return "SELL" 
            elif current_qty < 0:
                 # If target is zero or positive, we are buying to cover
                 # If target is still negative but less so, we are buying to cover partially
                 if target_qty >= 0 or target_qty > current_qty: 
                      return "BUY TO COVER" 
                 else: # Target is more negative, selling more short
                      return "SELL SHORT" 
            else: # current_qty is 0, target must be negative (since net_qty < 0)
                 return "SELL SHORT" 
        else: # net_qty == 0
            return "NONE"

    merged_df['Action'] = merged_df.apply(determine_action, axis=1)
    
    merged_df['Quantity'] = merged_df['Net Order Qty'].abs()

    # *** ADD: Determine Order Type ***
    def determine_order_type(row):
        target_qty = row['Ideal Target Qty']
        current_qty = row['Current Qty']
        if target_qty != 0 and current_qty != 0:
            return "ADJUST (Delta/Weighting)"
        elif target_qty != 0 and current_qty == 0:
            return "New"
        elif target_qty == 0 and current_qty != 0:
            return "Exit"
        else:
            return "None" # Should ideally not happen after filtering zero Qtys

    merged_df['Order Type'] = merged_df.apply(determine_order_type, axis=1)

    print(f"[DEBUG] Calculated Net Orders. Rows before filtering zero quantity: {len(merged_df)}")

    # --- Prepare Final Output DataFrame --- 
    final_orders_df = merged_df.copy()
    
    # Rename Ideal Target Qty back to Target Qty for output consistency
    final_orders_df = final_orders_df.rename(columns={'Ideal Target Qty': 'Target Qty'})

    # Rename other columns needed for final output 
    rename_map = {
         (weight_col if weight_col in final_orders_df.columns else 'Raw Weight'): 'Raw Weight Used',
         'Crncy': 'Currency',
         'Px Close': 'Local Px Close',
         'NormWeight': 'Normalized Weight Used' # Ensure NormWeight is renamed correctly
    }
    final_orders_df.rename(columns={k: v for k, v in rename_map.items() if k in final_orders_df.columns}, inplace=True)

    # Calculate final Actual USD Value based on Net Order Qty and Price Used
    final_orders_df['Actual USD Value'] = final_orders_df['Net Order Qty'] * final_orders_df['USD Price Used']

    # Define and select output columns, handle missing ones
    output_columns = [
        # *** ADD: Order Type, original_order (optional for final output?) ***
        'Order Type', 'Normalized Ticker', 'Schwab Symbol', 'Original Target Ticker', 'Action', 'Quantity',
        'Net Order Qty', 'Target Qty', 'Current Qty', 'USD Price Used', 'Actual USD Value',
        'Target USD', 'Security Type', 'Currency', 'FX Cls', 'Local Px Close',
        'Raw Weight Used', 'Normalized Weight Used', 'original_order' # Keep original_order for sorting
    ]
    # Ensure all needed columns exist, fill gaps with appropriate NA/0
    for col in output_columns:
         if col not in final_orders_df.columns:
              # More specific defaults
              if 'Qty' in col or 'Quantity' in col: default_val = 0
              elif 'Value' in col or 'USD' in col or 'Price' in col or 'FX' in col or 'Weight' in col: default_val = 0.0
              elif col == 'original_order': default_val = np.inf # Default exit orders to sort last
              else: default_val = ''
              final_orders_df[col] = default_val
              print(f"[DEBUG] Added missing output column '{col}' with default '{default_val}'")

    final_orders_df = final_orders_df[output_columns] # Select columns
    
    # *** MODIFY Sorting ***
    # Map Order Type for sorting: New/Adjust first, Exit last
    final_orders_df['Order Type Sort'] = final_orders_df['Order Type'].map({'New': 1, 'ADJUST (Delta/Weighting)': 1, 'Exit': 2, 'None': 3})
    # Sort by Order Type (Exit last), then by original Citrindex order
    final_orders_df = final_orders_df.sort_values(
        by=['Order Type Sort', 'original_order'],
        ascending=[True, True],
        na_position='last' # Should not have NaNs here, but good practice
    )
    # Remove temporary sort columns
    final_orders_df = final_orders_df.drop(columns=['Order Type Sort', 'original_order'])


    # Final summary stats: Calculate based on TARGET QTY column
    # Use Target Qty and Price Used for consistency
    valid_targets = final_orders_df[final_orders_df['Target Qty'] != 0] # Use the final Target Qty column
    final_target_long_val = (valid_targets.loc[valid_targets['Target Qty'] > 0, 'Target Qty'] * valid_targets['USD Price Used']).sum()
    final_target_short_val = abs((valid_targets.loc[valid_targets['Target Qty'] < 0, 'Target Qty'] * valid_targets['USD Price Used']).sum())
    final_target_gross_val = final_target_long_val + final_target_short_val # Calculated Gross
    final_net_target_val = final_target_long_val - final_target_short_val # Calculated Net

    print(f"\n[INFO] Final Target Portfolio Summary (based on TARGET quantities):")
    print(f"[INFO] Target Long Value    : ${final_target_long_val:,.2f}")
    print(f"[INFO] Target Short Value   : ${final_target_short_val:,.2f}")
    print(f"[INFO] Calculated Gross Value: ${final_target_gross_val:,.2f} (Target Gross Input: ${target_gross_exposure:,.2f})")
    print(f"[INFO] Calculated Net Value   : ${final_net_target_val:,.2f}")

    # --- Write Output Files --- 
    if output_file:
        # Create 'orders' directory if it doesn't exist
        os.makedirs("orders", exist_ok=True)
        excel_output_path = os.path.join("orders", output_file) # Place in orders dir
        
        # Filter out 'NONE' actions for the final output files, unless quantity is non-zero (shouldn't happen)
        orders_to_write = final_orders_df[final_orders_df['Action'] != 'NONE'].copy()
        # Alternatively, filter by Quantity != 0
        # orders_to_write = final_orders_df[final_orders_df['Quantity'] != 0].copy()
        print(f"[INFO] Total order lines generated (excluding NONE): {len(orders_to_write)}")

        if not orders_to_write.empty:
            print(f"[INFO] Writing detailed orders to Excel: {excel_output_path}")
            with pd.ExcelWriter(excel_output_path) as writer:
                 # Ensure the columns written are the intended ones (without sort keys)
                 cols_for_excel = [c for c in output_columns if c not in ['original_order']] # Exclude original_order from final file
                 orders_to_write[cols_for_excel].to_excel(writer, sheet_name="Net Orders", index=False)
            
            # Create Schwab-uploadable format 
            schwab_orders = orders_to_write.copy()
            
            # Use Schwab Symbol if available and not blank, otherwise use Normalized Ticker
            schwab_orders['Symbol'] = schwab_orders.apply(lambda row: row['Schwab Symbol'] if pd.notna(row['Schwab Symbol']) and row['Schwab Symbol'] != '' else row['Normalized Ticker'], axis=1)
            
            # Map actions for Schwab format
            schwab_orders['Schwab Action'] = schwab_orders['Action'].map({
                "BUY": "Buy", 
                "SELL": "Sell", 
                "SELL SHORT": "Sell Short",
                "BUY TO COVER": "Buy" # Schwab usually treats Buy to Cover as Buy
            })
            
            # Schwab Quantity is always positive
            schwab_orders['Schwab Quantity'] = schwab_orders['Quantity']
            
            # Select columns for the upload file
            schwab_upload = schwab_orders[['Symbol', 'Schwab Action', 'Schwab Quantity']].copy()
            schwab_upload.columns = ['Symbol', 'Action', 'Quantity'] # Rename for Schwab
            
            # Filter out rows with missing Action (shouldn't happen if filtered before) or zero Quantity
            schwab_upload = schwab_upload.dropna(subset=['Action'])
            schwab_upload = schwab_upload[schwab_upload['Quantity'] > 0]
            
            schwab_upload_path = os.path.join("orders", "schwab_upload.csv")
            print(f"[INFO] Writing Schwab upload format to CSV: {schwab_upload_path}")
            schwab_upload.to_csv(schwab_upload_path, index=False)

            # Also save the detailed net orders as CSV
            net_orders_csv_path = os.path.join("orders", "net_orders_detailed.csv")
            print(f"[INFO] Writing detailed net orders to CSV: {net_orders_csv_path}")
             # Ensure the columns written are the intended ones (without sort keys)
            cols_for_csv = [c for c in output_columns if c not in ['original_order']] # Exclude original_order from final file
            orders_to_write[cols_for_csv].to_csv(net_orders_csv_path, index=False)

            # *** ADD: Create Simplified Sanity Check CSV ***
            print(f"[INFO] Creating simplified sanity check CSV...")
            sanity_df = orders_to_write.copy() # Start with orders that have an action
            # Use Schwab Symbol if available, else Normalized Ticker
            sanity_df['Symbol'] = sanity_df.apply(lambda row: row['Schwab Symbol'] if pd.notna(row['Schwab Symbol']) and row['Schwab Symbol'] != '' else row['Normalized Ticker'], axis=1)
            # Simplify Order Type labels and make uppercase
            sanity_df['Status'] = sanity_df['Order Type'].replace({'ADJUST (Delta/Weighting)': 'ADJUST'}).str.upper()
            # Select and rename columns for the sanity check file
            sanity_check_cols = ['Symbol', 'Status', 'Action', 'Quantity']
            sanity_df_final = sanity_df[sanity_check_cols]
            
            sanity_check_path = os.path.join("orders", "sanity_check_orders.csv")
            print(f"[INFO] Writing simplified sanity check to CSV: {sanity_check_path}")
            sanity_df_final.to_csv(sanity_check_path, index=False)

        else:
            print("[WARNING] No non-NONE orders generated. Check input files and parameters.")
            # Still write the full dataframe to Excel for debugging maybe?
            print(f"[INFO] Writing full dataframe (including NONE) to Excel: {excel_output_path}")
            with pd.ExcelWriter(excel_output_path) as writer:
                 # Ensure the columns written are the intended ones (without sort keys)
                 cols_for_excel = [c for c in output_columns if c not in ['original_order']] # Exclude original_order from final file
                 final_orders_df[cols_for_excel].to_excel(writer, sheet_name="All Processed Rows", index=False)

    # Return the dataframe containing all rows (including NONE) for potential further inspection
     # Ensure the columns returned are the intended ones (without sort keys)
    cols_to_return = [c for c in output_columns if c not in ['original_order']]
    return final_orders_df[cols_to_return] 

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Citrindex portfolio, compare with current holdings, and generate net Schwab orders')
    parser.add_argument('input_file', help='Path to Citrindex portfolio Excel file')
    parser.add_argument('--target-gross-exposure', type=float, default=350000,
                        help='Target Gross Exposure (Longs + Abs(Shorts)) in USD')
    parser.add_argument('--output-file', default="schwab_orders.xlsx",
                        help='Base name for output Excel file (will be saved in ./orders/)')
    parser.add_argument('--min-allocation', type=float, default=10,
                        help='Minimum target allocation in USD to consider a position')
    parser.add_argument('--round-up', action='store_true', default=True,
                        help='Round up share quantities (default: True)')
    parser.add_argument('--no-round-up', action='store_false', dest='round_up',
                        help='Round down share quantities instead of up')
    parser.add_argument('--use-delta', action='store_true', default=False,
                        help='Use Delta Adj Wgt if available (default: False, uses Wgt or Market Value %%)')
    parser.add_argument('--us-only', action='store_true', default=False, # Default is False
                        help='Filter out non-US stocks *after* attempting replacements (default: False)')
    parser.add_argument('--find-us-replacements', action='store_true', default=False,
                        help='Try to find US replacements (ADRs/mapped tickers) for non-US stocks (default: False)')
    parser.add_argument('--update-prices', action='store_true', default=False,
                        help='Fetch current market prices using yfinance (default: False)')
    parser.add_argument('--current-portfolio-csv', 
                        help='Path to Schwab positions CSV file (e.g., Individual-Positions.csv) for calculating net orders')
    parser.add_argument('--exclude', nargs='+', default=[],
                        help='List of tickers to exclude from the target portfolio (e.g., --exclude CASH BRK.B)')


    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.input_file):
        print(f"Error: Target portfolio file '{args.input_file}' not found.")
        exit(1)
        
    if args.current_portfolio_csv and not os.path.exists(args.current_portfolio_csv):
         print(f"Error: Current portfolio file '{args.current_portfolio_csv}' not found.")
         exit(1)


    print("--- Starting Portfolio Processing ---")
    print(f"Target File: {args.input_file}")
    print(f"Current Holdings File: {args.current_portfolio_csv if args.current_portfolio_csv else 'N/A'}")
    print(f"Target Gross Exposure: ${args.target_gross_exposure:,.2f}")
    # Update print logic based on corrected defaults
    print(f"Find US Replacements: {args.find_us_replacements}") 
    print(f"Update Prices: {args.update_prices}") 
    print(f"US Only Filter: {args.us_only}")
    print(f"Rounding: {'Ceiling (Up)' if args.round_up else 'Floor (Down)'}")
    print(f"Min Allocation: ${args.min_allocation:,.2f}")
    print(f"Excluded Tickers: {args.exclude}")
    print("-" * 35)


    try:
        # Process the file
        final_orders = process_combined_sheet(
            input_file=args.input_file,
            target_gross_exposure=args.target_gross_exposure,
            output_file=args.output_file,
            exclude=args.exclude,
            use_delta=args.use_delta,
            min_allocation=args.min_allocation,
            round_up=args.round_up,
            us_only=args.us_only,
            find_us_replacements=args.find_us_replacements,
            update_prices=args.update_prices,
            current_portfolio_csv=args.current_portfolio_csv # Pass the new arg
        )
        
        if not final_orders.empty:
             print("--- Processing Complete ---")
             print(f"Generated {len(final_orders[final_orders['Action'] != 'NONE'])} net orders.")
             # Display summary of actions
             action_counts = final_orders['Action'].value_counts()
             print("Order Summary:")
             for action, count in action_counts.items():
                  print(f"- {action}: {count}")
             
             # Display summary of order types
             order_type_counts = final_orders['Order Type'].value_counts()
             print("Order Type Summary:")
             for otype, count in order_type_counts.items():
                 if otype != 'None': # Don't show 'None' type count
                     print(f"- {otype}: {count}")

             print(f"Output files saved in './orders/' directory.")
        else:
             print("--- Processing Complete ---")
             print("No orders were generated. Please check logs and input files.")

    except Exception as e:
        print("[--- An Error Occurred ---")
        import traceback
        traceback.print_exc() # Print detailed traceback
        # print(f"Error: {str(e)}") # Simpler error message
        exit(1)