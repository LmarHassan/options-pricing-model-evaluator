import datetime as dt
import pandas as pd
import yfinance as yf

def main():
    ticker = "MSFT"
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * 3 + 10)  # ~3 years + buffer

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        threads=False,
        group_by="column",
    )

    if df is None or df.empty:
        raise RuntimeError("No data returned from yfinance. Check internet/ticker.")

    df = df.reset_index()  # Date becomes a column
    df["Ticker"] = ticker

    # Keep standard columns (your fallback can use Close or Adj Close)
    wanted = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in wanted:
        if col not in df.columns:
            df[col] = pd.NA

    out = df[wanted].copy()

    # Ensure Date is YYYY-MM-DD (clean + consistent)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    out.to_csv("stocks.csv", index=False)
    print(f"Created stocks.csv with {len(out)} rows for {ticker} from {out['Date'].min()} to {out['Date'].max()}")

if __name__ == "__main__":
    main()
