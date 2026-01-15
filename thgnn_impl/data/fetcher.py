
import yfinance as yf
import pandas as pd
from .features import compute_features

def fetch_and_process_data(tickers, start_date, end_date):
    """
    Fetches data for tickers, computes features, and aligns them.
    Refactored from original data_loader.py
    """
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Download in bulk
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    
    df_dict = {}
    for ticker in tickers:
        try:
            # Handle multi-index columns if bulk download
            if len(tickers) > 1:
                df = data[ticker].copy()
            else:
                df = data.copy()
            
            if df.empty:
                print(f"Warning: No data for {ticker}")
                continue
                
            df_dict[ticker] = df
        except KeyError:
            print(f"Warning: Could not process {ticker}")
            continue

    # Add SPY if not in list for market features
    if 'SPY' not in tickers:
        try:
            spy = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)
            if not spy.empty:
                df_dict['SPY'] = spy
        except Exception as e:
            print(f"Warning: Could not fetch SPY: {e}")
        
    # Compute Features
    print("Computing features...")
    feature_dict = compute_features(df_dict)
    
    # Remove SPY from features if it wasn't in original list
    if 'SPY' not in tickers and 'SPY' in feature_dict:
        del feature_dict['SPY']
        
    # Align all dataframes to common index (intersection)
    print("Aligning data...")
    if not feature_dict:
        raise ValueError("No valid data fetched.")
        
    common_index = feature_dict[list(feature_dict.keys())[0]].index
    for t in feature_dict:
        common_index = common_index.intersection(feature_dict[t].index)
        
    aligned_data = {}
    for t, df in feature_dict.items():
        aligned_data[t] = df.loc[common_index]
        
    print(f"Data ready. Shape: {len(common_index)} days x {len(aligned_data)} stocks x {len(aligned_data[list(aligned_data.keys())[0]].columns)} features")
    return aligned_data
