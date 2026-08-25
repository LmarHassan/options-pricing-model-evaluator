import math
import datetime
import os
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf

EVALS_FILENAME = "evaluations.csv"


def _clean_ticker(ticker: str) -> str:
    if not isinstance(ticker, str):
        return ""
    s = ticker.strip().upper()
    s = "".join(c for c in s if c.isalnum() or c in ".-")
    return s


def _parse_date_yyyy_mm_dd(s: str) -> datetime.date:
    return datetime.datetime.strptime(str(s).strip(), "%Y-%m-%d").date()


def _time_fraction_days(days: int) -> float:
    return max(int(days), 1) / 365.0


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _ensure_datetime_index(idx) -> pd.DatetimeIndex:
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    return pd.to_datetime(idx, errors="coerce")


def _extract_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("yfinance returned no usable data frame.")

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" in lvl0:
            sub = df.xs("Close", level=0, axis=1)
        elif "Adj Close" in lvl0:
            sub = df.xs("Adj Close", level=0, axis=1)
        else:
            sub = df.xs(df.columns.get_level_values(0)[-1], level=0, axis=1)

        if isinstance(sub, pd.Series):
            series = sub
        else:
            if ticker in sub.columns:
                series = sub[ticker]
            elif sub.shape[1] == 1:
                series = sub.iloc[:, 0]
            else:
                series = sub.iloc[:, 0]
    else:
        if "Close" in df.columns:
            sub = df["Close"]
        elif "Adj Close" in df.columns:
            sub = df["Adj Close"]
        else:
            sub = df.iloc[:, -1]

        if isinstance(sub, pd.DataFrame):
            if sub.shape[1] == 1:
                series = sub.iloc[:, 0]
            else:
                if ticker in sub.columns:
                    series = sub[ticker]
                else:
                    series = sub.iloc[:, 0]
        else:
            series = sub

    series = pd.Series(series)
    series.index = _ensure_datetime_index(series.index)
    series = series.dropna()
    series = pd.to_numeric(series, errors="coerce").dropna()

    if series.empty:
        raise RuntimeError("Close series is empty after cleaning.")
    series = series.sort_index()

    if isinstance(series.index, pd.DatetimeIndex):
        try:
            if series.index.tz is not None:
                series.index = series.index.tz_convert(None)
        except Exception:
            try:
                series.index = series.index.tz_localize(None)
            except Exception:
                pass

    return series


def _fetch_prices_yfinance_download(ticker: str, start: datetime.date, end: datetime.date) -> pd.Series:
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        threads=False,
        group_by="column",
    )
    return _extract_close_series(df, ticker)


def _fetch_prices_yfinance_history(ticker: str, start: datetime.date, end: datetime.date) -> pd.Series:
    t = yf.Ticker(ticker)
    df = t.history(
        start=start.strftime("%Y-%m-%d"),
        end=(end + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        actions=False,
    )
    return _extract_close_series(df, ticker)


def _fetch_prices_local_csv(ticker: str, csv_path: str = "stocks.csv") -> pd.Series:
    if not os.path.exists(csv_path):
        raise RuntimeError("Local stocks.csv not found.")

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    if "date" not in cols:
        raise RuntimeError("stocks.csv missing a Date column.")
    date_col = cols["date"]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    if "ticker" in cols:
        tick_col = cols["ticker"]
        df = df[df[tick_col].astype(str).str.upper() == ticker]
        if df.empty:
            raise RuntimeError(f"No rows for ticker {ticker} in stocks.csv.")

    if "close" in cols:
        close_col = cols["close"]
    elif "adj close" in cols:
        close_col = cols["adj close"]
    else:
        numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            for c in df.columns:
                if c == date_col:
                    continue
                df[c] = pd.to_numeric(df[c], errors="coerce")
            numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise RuntimeError("stocks.csv has no usable numeric price column.")
        close_col = numeric_cols[-1]

    s = pd.Series(df[close_col].values, index=df[date_col].values)
    s.index = _ensure_datetime_index(s.index)
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s.sort_index()
    if s.empty:
        raise RuntimeError("stocks.csv series is empty after cleaning.")
    return s


class MathematicsFunctions:
    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    @staticmethod
    def compute_d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
        if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def compute_d2(d1: float, sigma: float, T: float) -> float:
        return d1 - sigma * math.sqrt(T)

    @staticmethod
    def historical_volatility(price_series: pd.Series, days_per_year: int = 252) -> float:
        if price_series is None:
            return 0.0
        if isinstance(price_series, pd.DataFrame):
            if price_series.shape[1] >= 1:
                price_series = price_series.iloc[:, 0]
            else:
                return 0.0
        s = pd.Series(price_series).dropna()
        if len(s) < 3:
            return 0.0
        returns = np.log(s / s.shift(1)).dropna()
        if len(returns) < 2:
            return 0.0
        v = float(returns.std(ddof=0) * math.sqrt(days_per_year))
        if not np.isfinite(v) or v <= 0:
            return 0.0
        return v


class Option:
    def __init__(self, S, K, r, T, sigma, option_type):
        self.S = float(S)
        self.K = float(K)
        self.r = float(r)
        self.T = max(float(T), 1e-8)
        self.sigma = max(float(sigma), 1e-8)
        self.option_type = option_type

    def price(self) -> float:
        raise NotImplementedError


class BlackScholes(Option):
    def price(self) -> float:
        d1 = MathematicsFunctions.compute_d1(self.S, self.K, self.r, self.sigma, self.T)
        d2 = MathematicsFunctions.compute_d2(d1, self.sigma, self.T)
        if self.option_type == "call":
            return self.S * MathematicsFunctions.norm_cdf(d1) - self.K * math.exp(-self.r * self.T) * MathematicsFunctions.norm_cdf(d2)
        return self.K * math.exp(-self.r * self.T) * MathematicsFunctions.norm_cdf(-d2) - self.S * MathematicsFunctions.norm_cdf(-d1)


class Black76(Option):
    def price(self) -> float:
        F = self.S * math.exp(self.r * self.T)
        d1 = (math.log(F / self.K) + 0.5 * self.sigma * self.sigma * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        df = math.exp(-self.r * self.T)
        if self.option_type == "call":
            return df * (F * MathematicsFunctions.norm_cdf(d1) - self.K * MathematicsFunctions.norm_cdf(d2))
        return df * (self.K * MathematicsFunctions.norm_cdf(-d2) - F * MathematicsFunctions.norm_cdf(-d1))


class BinomialCRR(Option):
    def __init__(self, S, K, r, T, sigma, option_type, steps=100):
        super().__init__(S, K, r, T, sigma, option_type)
        self.steps = max(1, int(steps))

    def price(self) -> float:
        dt = self.T / self.steps
        u = math.exp(self.sigma * math.sqrt(dt))
        d = 1.0 / u if u != 0 else 0.0
        a = math.exp(self.r * dt)
        denom = (u - d)
        if denom == 0 or not np.isfinite(denom):
            return 0.0
        p = (a - d) / denom
        p = min(max(p, 0.0), 1.0)

        prices = np.array([self.S * (u ** j) * (d ** (self.steps - j)) for j in range(self.steps + 1)], dtype=float)
        if self.option_type == "call":
            values = np.maximum(prices - self.K, 0.0)
        else:
            values = np.maximum(self.K - prices, 0.0)

        disc = math.exp(-self.r * dt)
        for _ in range(self.steps - 1, -1, -1):
            values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        return float(values[0])


class MonteCarloEuropean(Option):
    def __init__(self, S, K, r, T, sigma, option_type, num_paths=10000):
        super().__init__(S, K, r, T, sigma, option_type)
        self.num_paths = max(2000, int(num_paths))

    def price(self) -> float:
        Z = np.random.normal(size=self.num_paths)
        ST = self.S * np.exp((self.r - 0.5 * self.sigma * self.sigma) * self.T + self.sigma * math.sqrt(self.T) * Z)
        if self.option_type == "call":
            payoffs = np.maximum(ST - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - ST, 0.0)
        return float(math.exp(-self.r * self.T) * np.mean(payoffs))


class Bachelier(Option):
    def price(self) -> float:
        sigma_b = self.sigma * self.S
        if sigma_b <= 0 or self.T <= 0:
            d = 0.0
        else:
            d = (self.S - self.K) / (sigma_b * math.sqrt(self.T))
        df = math.exp(-self.r * self.T)
        if self.option_type == "call":
            return df * ((self.S - self.K) * MathematicsFunctions.norm_cdf(d) + sigma_b * math.sqrt(self.T) * MathematicsFunctions.norm_pdf(d))
        return df * ((self.K - self.S) * MathematicsFunctions.norm_cdf(-d) + sigma_b * math.sqrt(self.T) * MathematicsFunctions.norm_pdf(-d))


class BinaryOption(Option):
    def price(self) -> float:
        d1 = MathematicsFunctions.compute_d1(self.S, self.K, self.r, self.sigma, self.T)
        d2 = MathematicsFunctions.compute_d2(d1, self.sigma, self.T)
        if self.option_type == "call":
            return math.exp(-self.r * self.T) * MathematicsFunctions.norm_cdf(d2)
        return math.exp(-self.r * self.T) * MathematicsFunctions.norm_cdf(-d2)


class AsianOptionArithmetic(Option):
    def __init__(self, S, K, r, T, sigma, option_type, num_paths=5000, fixings=50):
        super().__init__(S, K, r, T, sigma, option_type)
        self.num_paths = max(2000, int(num_paths))
        self.fixings = max(2, int(fixings))

    def price(self) -> float:
        dt = self.T / self.fixings
        Z = np.random.normal(size=(self.num_paths, self.fixings))
        paths = np.zeros((self.num_paths, self.fixings), dtype=float)
        paths[:, 0] = self.S * np.exp((self.r - 0.5 * self.sigma * self.sigma) * dt + self.sigma * math.sqrt(dt) * Z[:, 0])
        for t in range(1, self.fixings):
            paths[:, t] = paths[:, t - 1] * np.exp((self.r - 0.5 * self.sigma * self.sigma) * dt + self.sigma * math.sqrt(dt) * Z[:, t])
        avg = paths.mean(axis=1)
        if self.option_type == "call":
            payoffs = np.maximum(avg - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - avg, 0.0)
        return float(math.exp(-self.r * self.T) * np.mean(payoffs))


def build_model_class_map():
    return {
        "BlackScholes": BlackScholes,
        "Black76": Black76,
        "BinomialCRR": BinomialCRR,
        "MonteCarloEuropean": MonteCarloEuropean,
        "Bachelier": Bachelier,
        "BinaryOption": BinaryOption,
        "AsianOptionArithmetic": AsianOptionArithmetic,
    }


class AutomatedInputs:
    def __init__(self, ticker: str, expiry: datetime.date, r: float):
        self.ticker = _clean_ticker(ticker)
        self.expiry = expiry
        self.r = float(r)
        self.historical_prices: Optional[pd.Series] = None
        self.current_price: Optional[float] = None
        self.volatility: Optional[float] = None

    def historical_data(self) -> None:
        if not self.ticker:
            raise RuntimeError("Ticker is blank.")

        end = datetime.date.today()
        days_to_expiry = max((self.expiry - end).days, 1)
        lookback = max(days_to_expiry, 60) + 10
        start = end - datetime.timedelta(days=lookback * 2)

        last_err: Optional[Exception] = None

        for fetcher in (_fetch_prices_yfinance_download, _fetch_prices_yfinance_history):
            for _ in range(2):
                try:
                    s = fetcher(self.ticker, start, end)

                    first_date = s.index.min().date() if hasattr(s.index, "min") else "?"
                    last_date2 = s.index.max().date() if hasattr(s.index, "max") else "?"
                    print(f"[OBJ3] SOURCE=YFINANCE | ticker={self.ticker} | rows={len(s)} | range={first_date}→{last_date2} | fetcher={fetcher.__name__}")

                    self.historical_prices = s
                    self.current_price = float(s.iloc[-1])
                    return
                except Exception as e:
                    last_err = e

        try:
            s = _fetch_prices_local_csv(self.ticker)

            first_date = s.index.min().date() if hasattr(s.index, "min") else "?"
            last_date2 = s.index.max().date() if hasattr(s.index, "max") else "?"
            print(f"[OBJ3] SOURCE=LOCAL_CSV | ticker={self.ticker} | rows={len(s)} | range={first_date}→{last_date2} | file=stocks.csv")

            self.historical_prices = s
            self.current_price = float(s.iloc[-1])
            return
        except Exception as e:
            if last_err is None:
                raise RuntimeError(f"Unable to retrieve price data for {self.ticker}.") from e
            raise RuntimeError(
                f"Unable to retrieve price data for {self.ticker}. "
                f"yfinance error: {last_err}. Also failed local fallback."
            ) from e

    def compute_volatility(self) -> None:
        if self.historical_prices is None:
            self.historical_data()
        vol = MathematicsFunctions.historical_volatility(self.historical_prices)
        self.volatility = float(vol) if vol and vol > 0 else 0.2

    def as_package(self) -> Dict[str, Any]:
        if self.historical_prices is None:
            self.historical_data()
        if self.volatility is None:
            self.compute_volatility()
        return {
            "historical_prices": self.historical_prices,
            "current_price": self.current_price,
            "volatility": self.volatility,
            "expiry": self.expiry,
            "r": self.r,
        }


class ModelEvaluator:
    def __init__(self, manual_inputs: Dict[str, Any], auto_pkg: Dict[str, Any], model_classes: Dict[str, Any]):
        self.manual = manual_inputs
        self.auto = auto_pkg
        self.model_classes = model_classes
        self.historical_slice: Optional[pd.Series] = None
        self.results: Dict[str, Dict[str, float]] = {}
        self.accepted: List[str] = []

    @staticmethod
    def determine_delta(days_to_expiry: int) -> int:
        if days_to_expiry <= 7:
            return 1
        if days_to_expiry <= 30:
            return 3
        return 5

    def create_historical_slice(self) -> None:
        prices = pd.Series(self.auto["historical_prices"]).dropna()
        days_to_expiry = max((self.auto["expiry"] - datetime.date.today()).days, 1)
        slice_len = days_to_expiry
        if len(prices) < slice_len + 10:
            self.historical_slice = prices
        else:
            self.historical_slice = prices.iloc[-slice_len - 10 : -10]

        hs = self.historical_slice
        if hs is not None and len(hs) > 0:
            first_date = hs.index.min().date() if hasattr(hs.index, "min") else "?"
            last_date = hs.index.max().date() if hasattr(hs.index, "max") else "?"
            print(f"[OBJ3] SLICE_CREATED | rows={len(hs)} | range={first_date}→{last_date} | slice_len_target={slice_len}")

    def evaluate_models(self) -> None:
        self.create_historical_slice()
        prices = self.historical_slice
        if prices is None or len(prices) < 10:
            raise RuntimeError("Not enough historical data to evaluate models.")

        dates = prices.index.to_pydatetime()
        errors: Dict[str, List[float]] = {name: [] for name in self.model_classes.keys()}
        delta_days = self.determine_delta(max((self.auto["expiry"] - datetime.date.today()).days, 1))

        for i in range(len(prices) - delta_days):
            date_t = dates[i].date()
            S_t = float(prices.iloc[i])
            S_future = float(prices.iloc[i + delta_days])
            days_until_expiry_at_t = max((self.auto["expiry"] - date_t).days, 1)
            T_t = _time_fraction_days(days_until_expiry_at_t)

            sigma_t = MathematicsFunctions.historical_volatility(prices.iloc[: i + 1]) if i >= 2 else 0.0
            if sigma_t <= 0:
                sigma_t = float(self.auto["volatility"])

            for name, cls in self.model_classes.items():
                try:
                    model = cls(S_t, self.manual["K"], self.manual["r"], T_t, sigma_t, self.manual["option_type"])
                    predicted = float(model.price())
                    if not np.isfinite(predicted):
                        predicted = 0.0
                except Exception:
                    predicted = 0.0

                if self.manual["option_type"] == "call":
                    intrinsic = max(S_future - self.manual["K"], 0.0)
                else:
                    intrinsic = max(self.manual["K"] - S_future, 0.0)

                actual = intrinsic * math.exp(-self.manual["r"] * _time_fraction_days(delta_days))
                errors[name].append((predicted - actual) ** 2)

        avg_actuals = []
        for i in range(len(prices) - delta_days):
            S_future = float(prices.iloc[i + delta_days])
            intrinsic = max(S_future - self.manual["K"], 0.0) if self.manual["option_type"] == "call" else max(self.manual["K"] - S_future, 0.0)
            avg_actuals.append(intrinsic * math.exp(-self.manual["r"] * _time_fraction_days(delta_days)))
        avg_actual_price = float(np.mean(avg_actuals)) if avg_actuals else 0.0

        threshold_abs = (self.manual.get("error_threshold_percent", 10.0) / 100.0) * (avg_actual_price if avg_actual_price > 0 else 1.0)

        metrics: Dict[str, Dict[str, float]] = {}
        for name, sqerrs in errors.items():
            if not sqerrs:
                mse = float("inf")
                rmse = float("inf")
            else:
                mse = float(np.mean(sqerrs))
                rmse = float(math.sqrt(mse))
            metrics[name] = {"rmse": rmse, "mse": mse, "count": float(len(sqerrs))}

        self.results = metrics
        self.accepted = [n for n, m in metrics.items() if np.isfinite(m["rmse"]) and m["rmse"] <= threshold_abs]

        if not self.accepted:
            finite = [(n, m) for n, m in metrics.items() if np.isfinite(m["rmse"])]
            if finite:
                self.accepted = [sorted(finite, key=lambda x: x[1]["rmse"])[0][0]]
            else:
                self.accepted = list(self.model_classes.keys())[:1]

    def compute_today_predictions(self) -> Dict[str, Any]:
        accepted_models = list(self.accepted)
        preds: Dict[str, float] = {}
        weights: Dict[str, float] = {}

        S_today = float(self.auto["current_price"])
        days_to_expiry = max((self.auto["expiry"] - datetime.date.today()).days, 1)
        T_today = _time_fraction_days(days_to_expiry)
        sigma = float(self.auto["volatility"])

        for name in accepted_models:
            cls = self.model_classes[name]
            try:
                model = cls(S_today, self.manual["K"], self.manual["r"], T_today, sigma, self.manual["option_type"])
                p = float(model.price())
                if not np.isfinite(p):
                    p = 0.0
            except Exception:
                p = 0.0

            preds[name] = p
            rmse = float(self.results.get(name, {}).get("rmse", 1e-8))
            weights[name] = 1.0 / max(rmse, 1e-8)

        total_weight = sum(weights.values()) if weights else 1.0
        weighted_price = sum(preds[n] * weights[n] for n in preds) / total_weight

        finite_results = {k: v for k, v in self.results.items() if np.isfinite(v.get("rmse", np.inf))}
        best_model = min(finite_results.items(), key=lambda x: x[1]["rmse"])[0] if finite_results else None
        single_best_price = preds.get(best_model) if best_model else None

        return {
            "accepted_models": accepted_models,
            "preds": preds,
            "weights": weights,
            "weighted_price": float(weighted_price),
            "best_model": best_model,
            "single_best_price": float(single_best_price) if single_best_price is not None else None,
        }

    def recommendation(self, final_price: float) -> str:
        market = float(self.manual["current_option_price"])
        if final_price > market:
            return "THE PRICE OF THE OPTION IS EXPECTED TO RISE. THE SYSTEM RECOMMENDS YOU TO BUY THE OPTION."
        if final_price < market:
            return "THE PRICE OF THE OPTION IS EXPECTED TO FALL. THE SYSTEM RECOMMENDS YOU TO NOT BUY THE OPTION."
        return "THE PRICE OF THE OPTION IS EXPECTED TO REMAIN THE SAME. THE SYSTEM RECOMMENDS YOU TO HOLD."


def append_evaluation_row(row: Dict[str, Any], filename: str = EVALS_FILENAME) -> None:
    df_row = pd.DataFrame([row])
    if os.path.exists(filename):
        try:
            df_existing = pd.read_csv(filename, parse_dates=["timestamp"])
            df_combined = pd.concat([df_existing, df_row], ignore_index=True)
            df_combined.to_csv(filename, index=False)
            return
        except Exception:
            df_row.to_csv(filename, index=False)
            return
    df_row.to_csv(filename, index=False)


def read_evaluations(filename: str = EVALS_FILENAME) -> pd.DataFrame:
    if not os.path.exists(filename):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filename, parse_dates=["timestamp"])
        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def run_full_evaluation(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ticker = _clean_ticker(payload.get("ticker", ""))
    option_type = (payload.get("option_type", "") or "").strip().lower()
    if option_type not in ("call", "put"):
        raise RuntimeError("Option type must be 'call' or 'put'.")

    try:
        K = float(payload.get("K"))
        if K <= 0:
            raise RuntimeError("Strike price must be > 0.")
    except Exception:
        raise RuntimeError("Invalid strike price.")

    try:
        expiry = _parse_date_yyyy_mm_dd(payload.get("expiry_date", ""))
    except Exception:
        raise RuntimeError("Expiry date must be in format YYYY-MM-DD.")
    if expiry <= datetime.date.today():
        raise RuntimeError("Expiry date must be in the future.")

    try:
        market_price = float(payload.get("current_option_price", 0))
        if market_price < 0:
            raise RuntimeError("Market option price must be >= 0.")
    except Exception:
        raise RuntimeError("Invalid market option price.")

    try:
        r_pct = float(payload.get("risk_free_r_pct", 0.5))
        r = r_pct / 100.0
        if r < -0.5 or r > 1.0:
            raise RuntimeError("Risk-free rate must be entered as a percent like 0.5 for 0.5%.")
    except Exception:
        raise RuntimeError("Invalid risk-free rate percent.")

    try:
        err_thr = float(payload.get("error_threshold_percent", 10.0))
        if err_thr <= 0:
            raise RuntimeError("Error threshold must be > 0.")
    except Exception:
        raise RuntimeError("Invalid error threshold percent.")

    manual = {
        "ticker": ticker,
        "option_type": option_type,
        "K": float(K),
        "expiry_date": expiry,
        "current_option_price": float(market_price),
        "r": float(r),
        "error_threshold_percent": float(err_thr),
    }

    auto_layer = AutomatedInputs(ticker=ticker, expiry=expiry, r=r)
    auto_layer.historical_data()
    auto_layer.compute_volatility()
    auto_pkg = auto_layer.as_package()

    model_map = build_model_class_map()
    evaluator = ModelEvaluator(manual, auto_pkg, model_map)
    evaluator.evaluate_models()
    preds = evaluator.compute_today_predictions()

    final_price = float(preds["weighted_price"])
    recommendation = evaluator.recommendation(final_price)

    row = {
        "timestamp": pd.Timestamp.now(),
        "ticker": ticker,
        "option_type": option_type,
        "strike": float(K),
        "expiry_date": expiry.isoformat(),
        "current_option_price": float(market_price),
        "risk_free_r": float(r),
        "volatility_used": float(auto_pkg.get("volatility", np.nan)),
        "accepted_models": "|".join(preds["accepted_models"]) if preds.get("accepted_models") else "",
        "weighted_price": float(preds.get("weighted_price", np.nan)),
        "best_model": preds.get("best_model", "") or "",
        "single_best_price": float(preds.get("single_best_price", np.nan)) if preds.get("single_best_price") is not None else np.nan,
        "recommendation": recommendation,
    }
    append_evaluation_row(row, EVALS_FILENAME)

    results = {
        "ticker": ticker,
        "option_type": option_type,
        "strike": float(K),
        "expiry_date": expiry.isoformat(),
        "market_option_price": float(market_price),
        "risk_free_rate": float(r),
        "spot_price": float(auto_pkg.get("current_price", np.nan)),
        "volatility": float(auto_pkg.get("volatility", np.nan)),
        "accepted_models": preds["accepted_models"],
        "model_prices": preds["preds"],
        "weighted_price": final_price,
        "best_model": preds["best_model"],
        "single_best_price": preds["single_best_price"],
        "recommendation": recommendation,
    }

    diagnostics = {"rmse_table": evaluator.results}
    return results, diagnostics
