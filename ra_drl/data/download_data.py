
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from tqdm import tqdm
from config import DOW30_TICKERS, TRAIN_START, TEST_END, DATA_DIR


def download_ticker(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=False
    )

    if df.empty:
        raise ValueError("Empty dataframe. No data is returned from Yahoo Finance")

    # fixing multiIndex problem
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # lowercase column names
    df.columns = [c.lower() for c in df.columns]

    df["ticker"] = ticker

    return df


def download_all(tickers, start, end):
    all_data = {}

    for ticker in tqdm(tickers):
        try:
            df = download_ticker(ticker, start, end)
            if not df.empty:
                all_data[ticker] = df
        except Exception as e:
            print(f"Failed {ticker} -> {e}")

    return all_data



# Converts the downloaded data into a matrix where:

# Rows → dates
# Columns → tickers
# Values → price (or volume) for that column

def build_price_matrix(all_data, column="close"):
    frames = {}

    for ticker, df in all_data.items():
        frames[ticker] = df[column]

    price_matrix = pd.DataFrame(frames)
    price_matrix = price_matrix.ffill().dropna()

    return price_matrix


def main():

    os.makedirs(DATA_DIR, exist_ok=True)

    all_data = download_all(DOW30_TICKERS, TRAIN_START, TEST_END)

    close_prices = build_price_matrix(all_data, "close")
    open_prices  = build_price_matrix(all_data, "open")
    high_prices  = build_price_matrix(all_data, "high")
    low_prices   = build_price_matrix(all_data, "low")
    volume       = build_price_matrix(all_data, "volume")

    close_prices.to_csv(os.path.join(DATA_DIR, "close_prices.csv"))
    open_prices.to_csv(os.path.join(DATA_DIR, "open_prices.csv"))
    high_prices.to_csv(os.path.join(DATA_DIR, "high_prices.csv"))
    low_prices.to_csv(os.path.join(DATA_DIR, "low_prices.csv"))
    volume.to_csv(os.path.join(DATA_DIR, "volume.csv"))

    print("Data saved successfully!")


if __name__ == "__main__":
    main()
