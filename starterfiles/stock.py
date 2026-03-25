import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# STUDENT CHANGE LOG & AI DISCLOSURE:
# 1. Did you use an LLM (ChatGPT/Claude/etc.)? [Yes]
# 2. If yes, what was your primary prompt?
#    "Step-by-step help to complete Stock class methods: get_data, calc_returns,
#     add_technical_indicators, plot_return_dist, plot_performance, and a test main()."

DEFAULT_START = dt.date.isoformat(dt.date.today() - dt.timedelta(365))
DEFAULT_END = dt.date.isoformat(dt.date.today())


class Stock:
    def __init__(self, symbol, start=DEFAULT_START, end=DEFAULT_END):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = self.get_data()

    def get_data(self):
        """
        Downloads data from yfinance and triggers return calculation.
        - Uses a free stock data API (yfinance)
        - Stores results in a pandas DataFrame
        - Index is date and converted to pandas Datetime
        """
        # yfinance treats 'end' as exclusive; add 1 day to include the end date
        end_dt = pd.to_datetime(self.end) + pd.Timedelta(days=1)

        df = yf.download(
            self.symbol,
            start=self.start,
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False
        )

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for symbol '{self.symbol}' "
                f"from {self.start} to {self.end}."
            )

        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        # Enrich with returns columns
        self.calc_returns(df)

        return df

    def calc_returns(self, df):
        """
        Adds 'change' and 'instant_return' columns to the dataframe.
        Requirement: Use vectorized pandas operations, not loops.

        change:
            difference in Close vs previous day's Close
        instant_return:
            daily instantaneous return = log(Close).diff().round(4)
        """
        if "Close" not in df.columns:
            raise ValueError("Expected 'Close' column in downloaded data.")

        df["change"] = df["Close"].diff()
        df["instant_return"] = np.log(df["Close"]).diff().round(4)

    def add_technical_indicators(self, windows=(20, 50)):
        """
        Add Simple Moving Averages (SMA) for the given windows to the internal DataFrame.
        Produce a plot showing the closing price and SMAs.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data found. Call get_data() first.")

        for w in windows:
            self.data[f"sma_{w}"] = self.data["Close"].rolling(window=w).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data["Close"], label="Close")

        for w in windows:
            plt.plot(self.data.index, self.data[f"sma_{w}"], label=f"SMA {w}")

        plt.title(f"{self.symbol} Close Price with Moving Averages")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_return_dist(self, bins=50):
        """
        Plot a well formatted histogram of the instantaneous returns.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data found. Call get_data() first.")
        if "instant_return" not in self.data.columns:
            raise ValueError("Missing 'instant_return'. Run get_data() first.")

        rets = self.data["instant_return"].dropna()

        plt.figure(figsize=(10, 6))
        plt.hist(rets, bins=bins)
        plt.title(f"{self.symbol} Instantaneous Return Distribution")
        plt.xlabel("Instantaneous Return (log)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_performance(self):
        """
        Plot a well formatted line graph of the stock’s performance over the range of data collected,
        as a percent gain/loss relative to the first Close.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data found. Call get_data() first.")

        base = float(self.data["Close"].iloc[0])
        perf_pct = (self.data["Close"] / base - 1.0) * 100.0

        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, perf_pct)
        plt.title(f"{self.symbol} Performance (% Gain/Loss)")
        plt.xlabel("Date")
        plt.ylabel("Performance (%)")
        plt.axhline(0, linewidth=1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    # Instantiate a test object
    aapl = Stock("AAPL")

    # Access the data attribute
    print(aapl.data.head())
    print("\nColumns:", list(aapl.data.columns))

    # Generate the two plots
    aapl.plot_return_dist()
    aapl.plot_performance()

    # Optional extra plot (technical indicators)
    aapl.add_technical_indicators()


if __name__ == "__main__":
    main()