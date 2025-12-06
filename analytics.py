import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from typing import Tuple, Dict

def resample_bars(df_ticks: pd.DataFrame, rule: str = "1s") -> pd.DataFrame:
    """
    Resample tick data into OHLCV bars per symbol.
    df_ticks: DataFrame indexed by timestamp and with columns: symbol, price, size (optional)
    Returns a DataFrame with index 'timestamp' and columns: open, high, low, close, volume, symbol
    """
    if df_ticks is None or df_ticks.empty:
        return pd.DataFrame()

    out_frames = []
    # ensure index is datetime-like
    if not pd.api.types.is_datetime64_any_dtype(df_ticks.index):
        try:
            df_ticks = df_ticks.copy()
            df_ticks.index = pd.to_datetime(df_ticks.index, utc=True, errors='coerce')
            df_ticks = df_ticks.dropna(subset=[df_ticks.index.name or df_ticks.index])
        except Exception:
            pass

    for sym, g in df_ticks.groupby("symbol"):
        if g.empty:
            continue
        g = g.sort_index()

        # produce OHLC using agg to avoid possible deprecations
        ohlc = g['price'].resample(rule).agg(
            open='first', high='max', low='min', close='last'
        )

        # volume: use size if available otherwise count of ticks as proxy
        if 'size' in g.columns:
            vol = g['size'].resample(rule).sum().rename("volume")
        else:
            vol = g['price'].resample(rule).count().rename("volume")

        # combine
        # ohlc is a DataFrame with columns open/high/low/close and index = timestamp
        ohlc = ohlc.assign(volume=vol, symbol=sym)
        ohlc = ohlc.dropna(subset=['close'])
        if not ohlc.empty:
            out_frames.append(ohlc)

    if not out_frames:
        return pd.DataFrame()

    bars = pd.concat(out_frames)
    # Ensure timestamp index name
    if 'timestamp' not in bars.index.names:
        bars.index.name = "timestamp"
    bars = bars.reset_index().set_index('timestamp')
    return bars

def compute_ols_hedge(series_a: pd.Series, series_b: pd.Series) -> Tuple[float, float, float]:
    """
    OLS regression of A ~ B. Returns (beta, intercept, r2).
    If insufficient samples, returns (1.0, 0.0, 0.0) as safe defaults.
    """
    df = pd.concat([series_a, series_b], axis=1).dropna()
    if df.shape[0] < 5:
        return 1.0, 0.0, 0.0
    y = df.iloc[:, 0].astype(float)
    X = df.iloc[:, 1].astype(float)
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    # defensive access to params
    try:
        beta = float(model.params[1])
    except Exception:
        beta = 1.0
    try:
        intercept = float(model.params[0])
    except Exception:
        intercept = 0.0
    try:
        r2 = float(model.rsquared)
    except Exception:
        r2 = 0.0
    return beta, intercept, r2

def compute_spread_zscore(series_a: pd.Series, series_b: pd.Series, beta: float, rolling: int = 60) -> Tuple[pd.Series, pd.Series]:
    """
    Compute spread = A - beta * B and rolling z-score using a window.
    Returns (spread_series, zscore_series). If inputs are empty, returns empty Series.
    """
    if series_a is None or series_b is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    df = pd.concat([series_a, series_b], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    spread = df.iloc[:, 0].astype(float) - beta * df.iloc[:, 1].astype(float)
    roll_mean = spread.rolling(rolling).mean()
    roll_std = spread.rolling(rolling).std()
    # avoid division by zero by replacing 0 std with NaN
    roll_std = roll_std.replace(0, np.nan)
    z = (spread - roll_mean) / roll_std
    return spread, z

def run_adf(series: pd.Series) -> Dict:
    """
    Run Augmented Dickey-Fuller test and return a stable dict.
    Handles both 5- and 6-element returns from statsmodels.adfuller.
    """
    s = series.dropna()
    if len(s) < 20:
        return {"error": "not enough observations for ADF (need >= 20)", "nobs": int(len(s))}

    try:
        result = adfuller(s.values, maxlag=1, autolag=None)
    except Exception as e:
        return {"error": f"adfuller failed: {e}", "nobs": int(len(s))}

    # adfuller may return tuples of length 5 or 6 depending on statsmodels version.
    # Common shapes:
    # 5 -> (adfstat, pvalue, usedlag, nobs, critical_values)
    # 6 -> (adfstat, pvalue, usedlag, nobs, critical_values, icbest)
    out = {}
    try:
        adfstat = float(result[0])
        pvalue = float(result[1])
        usedlag = int(result[2])
        nobs = int(result[3])
        crit_values = result[4] if len(result) > 4 else {}
        icbest = float(result[5]) if len(result) > 5 else None

        out = {
            "test_stat": adfstat,
            "pvalue": pvalue,
            "usedlag": usedlag,
            "nobs": nobs,
            "crit_values": crit_values,
            "icbest": icbest
        }
    except Exception as e:
        out = {"error": f"unexpected adfuller return shape: {e}", "raw": str(result)}

    return out

def compute_rolling_correlation(series_a: pd.Series, series_b: pd.Series, window: int = 60) -> pd.Series:
    """
    Computes rolling correlation between two aligned price series. Returns pandas Series.
    """
    if series_a is None or series_b is None:
        return pd.Series(dtype=float)
    df = pd.concat([series_a, series_b], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    df.columns = ["A", "B"]
    rolling_corr = df["A"].rolling(window).corr(df["B"])
    return rolling_corr

def run_backtest(spread_series: pd.Series, zscore_series: pd.Series, entry_cutoff: float = 2.0, exit_cutoff: float = 0.0) -> Dict:
    """
    Simulate a mean-reversion strategy:
    - Long Spread if z < -entry_cutoff
    - Short Spread if z > entry_cutoff
    - Exit Long if z > -exit_cutoff (or z crosses 0)
    - Exit Short if z < exit_cutoff (or z crosses 0)
    
    Returns:
        {
            "trades": pd.DataFrame,
            "equity": pd.Series,
            "stats": dict
        }
    """
    if spread_series is None or zscore_series is None or spread_series.empty or zscore_series.empty:
        return {"error": "No data for backtest"}

    # Align data
    df = pd.concat([spread_series.rename("spread"), zscore_series.rename("zscore")], axis=1).dropna()
    if df.empty:
        return {"error": "No aligned data for backtest"}

    trades = []
    position = 0 # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_time = None
    equity_curve = [0.0] * len(df)
    current_equity = 0.0
    
    timestamps = df.index
    spreads = df["spread"].values
    zscores = df["zscore"].values

    for i in range(len(df)):
        ts = timestamps[i]
        price = spreads[i]
        z = zscores[i]
        
        # PnL logic if holding
        # Note: spread pnl = (price - entry) * pos
        # But we only realize on exit for simple trade list. 
        # For equity curve, we could mark-to-market, but let's just do realized pnl for simplicity/robustness match.
        
        # Check Exit
        if position == 1:
            # Long exit: mean reversion usually targets 0. 
            # Condition: z >= -exit_cutoff (e.g. z > -0.0 -> z > 0)
            if z >= -exit_cutoff: 
                pnl = price - entry_price
                current_equity += pnl
                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": ts,
                    "Type": "Long",
                    "Entry Price": entry_price,
                    "Exit Price": price,
                    "PnL": pnl
                })
                position = 0
        elif position == -1:
            # Short exit
            # Condition: z <= exit_cutoff (e.g. z < 0)
            if z <= exit_cutoff:
                pnl = entry_price - price
                current_equity += pnl
                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": ts,
                    "Type": "Short",
                    "Entry Price": entry_price,
                    "Exit Price": price,
                    "PnL": pnl
                })
                position = 0
        
        # Check Entry (if flat)
        if position == 0:
            if z < -entry_cutoff:
                position = 1
                entry_price = price
                entry_time = ts
            elif z > entry_cutoff:
                position = -1
                entry_price = price
                entry_time = ts
        
        equity_curve[i] = current_equity

    # Compile results
    df_trades = pd.DataFrame(trades)
    s_equity = pd.Series(equity_curve, index=timestamps)
    
    stats = {
        "Total Return": current_equity,
        "Trade Count": len(df_trades),
        "Win Rate": 0.0,
        "Max Drawdown": 0.0,
        "Avg PnL": 0.0
    }
    
    if not df_trades.empty:
        stats["Win Rate"] = len(df_trades[df_trades["PnL"] > 0]) / len(df_trades)
        stats["Avg PnL"] = df_trades["PnL"].mean()
        
        # Simple Max DD on realized equity curve
        cum_max = s_equity.cummax()
        # Avoid division by zero or weirdness if equity is near 0. 
        # Typically DD is % from peak, but here we have absolute PnL on spread units.
        # We'll just show max absolute drawdown from peak.
        drawdown = s_equity - cum_max
        stats["Max Drawdown"] = drawdown.min()

    return {
        "trades": df_trades,
        "equity": s_equity,
        "stats": stats
    }

def compute_kalman_hedge(series_a: pd.Series, series_b: pd.Series, delta: float = 1e-4) -> Tuple[float, float, pd.Series, pd.Series]:
    """
    Estimate dynamic hedge ratio using Kalman Filter.
    Model: y = alpha + beta * x
    Returns: (latest_beta, latest_intercept, series_beta, series_intercept)
    """
    if series_a is None or series_b is None:
        return 1.0, 0.0, pd.Series(dtype=float), pd.Series(dtype=float)
        
    df = pd.concat([series_a, series_b], axis=1).dropna()
    if df.empty:
        return 1.0, 0.0, pd.Series(dtype=float), pd.Series(dtype=float)
        
    y = df.iloc[:, 0].values # Series A (dependent)
    x = df.iloc[:, 1].values # Series B (independent)
    n = len(y)
    
    # State: [alpha, beta]
    # x_t = F x_{t-1} + w_t,  w ~ N(0, Q)
    # y_t = H_t x_t + v_t,    v ~ N(0, R)
    
    # Initialization
    state_mean = np.zeros(2) # [intercept, slope]
    state_cov = np.ones((2, 2)) * 1.0
    
    # Random walk transitions (F=I)
    F = np.eye(2)
    
    # Process noise covariance (Q)
    # Allows coefficients to drift. 'delta' controls flexibility.
    Q = np.eye(2) * delta
    
    # Measurement noise variance (R)
    R = 1.0 
    
    betas = np.zeros(n)
    alphas = np.zeros(n)
    
    for t in range(n):
        # 1. Prediction Step
        # x_{t|t-1} = F * x_{t-1|t-1}
        # P_{t|t-1} = F * P_{t-1|t-1} * F^T + Q
        pred_state = state_mean # since F is I
        pred_cov = state_cov + Q
        
        # 2. Update Step
        # H_t = [1, x_t]
        H = np.array([1.0, x[t]])
        
        # Innovation/Residual
        # y_pred = H * pred_state
        y_pred = np.dot(H, pred_state)
        error = y[t] - y_pred
        
        # Innovation Covariance S = H P H^T + R
        S = np.dot(H, np.dot(pred_cov, H.T)) + R
        
        # Kalman Gain K = P H^T S^-1
        K = np.dot(pred_cov, H.T) / S 
        
        # New State
        # x_{t|t} = x_{t|t-1} + K * error
        state_mean = pred_state + K * error
        
        # New Covariance
        # P_{t|t} = (I - K H) P_{t|t-1} 
        # (Using Joseph form is more stable but simple form usually fine here)
        state_cov = pred_cov - np.outer(K, H).dot(pred_cov)
        
        alphas[t] = state_mean[0]
        betas[t]  = state_mean[1]
        
    return state_mean[1], state_mean[0], pd.Series(betas, index=df.index), pd.Series(alphas, index=df.index)

def compute_microstructure_metrics(df_ticks: pd.DataFrame, window_seconds: int = 60) -> pd.DataFrame:
    """
    Compute liquidity/microstructure metrics over the last window_seconds:
    - Quote Volume (Price * Size)
    - Trade Volume (Sum Size)
    - Tick Arrival Rate (ticks/sec)
    """
    if df_ticks is None or df_ticks.empty:
        return pd.DataFrame()
        
    # Ensure datetime index
    df = df_ticks.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        # try to set index from timestamp column if exists
        if "timestamp" in df.columns:
            # Use mixed format to handle potential differences (e.g. some with micros, some without)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
            df = df.set_index("timestamp")
        else:
            return pd.DataFrame()
            
    # Filter by time window
    cutoff = df.index.max() - pd.Timedelta(seconds=window_seconds)
    mask = df.index >= cutoff
    recent = df.loc[mask]
    
    if recent.empty:
        return pd.DataFrame()
        
    metrics = []
    for sym, g in recent.groupby("symbol"):
        count = len(g)
        # arrival rate = count / window (or time span if < window, but standard is window)
        arrival_rate = count / window_seconds
        
        trade_vol = 0.0
        quote_vol = 0.0
        
        if "size" in g.columns:
            trade_vol = g["size"].sum()
            if "price" in g.columns:
                quote_vol = (g["price"] * g["size"]).sum()
        elif "price" in g.columns:
            # Fallback if no size: treat size=1? Or just 0
            # Usually better to report 0 or count
            trade_vol = count
            quote_vol = g["price"].sum()
            
        metrics.append({
            "symbol": sym,
            "Arrival Rate (Hz)": arrival_rate,
            "Trade Vol (Qty)": trade_vol,
            "Quote Vol ($)": quote_vol
        })
        
    return pd.DataFrame(metrics).set_index("symbol")

def compute_correlation_matrix(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise correlation of close prices for all symbols in bars.
    """
    if bars.empty:
        return pd.DataFrame()
    
    # Pivot to get close prices table: index=timestamp, columns=symbol
    pivot = bars.pivot_table(index="timestamp", columns="symbol", values="close")
    if pivot.empty:
        return pd.DataFrame()
        
    return pivot.corr()

def compute_volatility(bars: pd.DataFrame, window: int = 60, timeframe: str = "1s") -> pd.DataFrame:
    """
    Compute rolling annualized volatility.
    Factor depends on timeframe. Default is high freq approximation.
    Returns DataFrame with columns [symbol, timestamp, volatility].
    """
    if bars.empty:
        return pd.DataFrame()
        
    # Determine annualization factor
    # Crypto 24/7: 365 days * 24 hours * 60 minutes * 60 seconds = 31,536,000 seconds
    # If timeframe is '1s', factor is sqrt(31,536,000) ~ 5615
    # If timeframe is '1min', factor is sqrt(31,536,000 / 60) = sqrt(525,600) ~ 725
    
    factor = 1.0
    if timeframe == "1s":
        factor = np.sqrt(31536000)
    elif timeframe == "1min":
        factor = np.sqrt(525600)
    elif timeframe == "5min":
        factor = np.sqrt(105120)
    else:
        # Default fallback to 1s-like scaling or just raw std
        factor = 1.0
        
    out_list = []
    
    for sym, g in bars.groupby("symbol"):
        g = g.sort_index()
        # pct_change gives returns
        rets = g["close"].pct_change()
        # rolling std of returns * annualized factor
        vol = rets.rolling(window).std() * factor
        
        # reconstruct df
        temp = pd.DataFrame({
            "volatility": vol,
            "symbol": sym
        })
        out_list.append(temp.dropna())
        
    if not out_list:
        return pd.DataFrame()
        
    return pd.concat(out_list).sort_index()

def compute_risk_metrics(spread: pd.Series, timeframe: str = "1s") -> dict:
    """
    Compute risk and alpha metrics for the spread series:
    - Half-life (Mean Reversion Speed)
    - Hurst Exponent (Trend vs Reversion)
    - Sharpe Ratio (Theoretical)
    """
    if spread is None or spread.empty or len(spread) < 20:
        return {"Half-Life": np.nan, "Hurst": np.nan, "Sharpe": 0.0}

    # 1. Half-Life
    # Delta S_t = -lambda * (S_{t-1} - mu) + epsilon
    # We run OLS: Delta S_t ~ S_{t-1} + const
    # lambda = -slope
    # HL = ln(2) / lambda
    try:
        s_lag = spread.shift(1).dropna()
        s_delta = spread.diff().dropna()
        # Align
        idx = s_lag.index.intersection(s_delta.index)
        s_lag = s_lag.loc[idx]
        s_delta = s_delta.loc[idx]
        
        if len(idx) > 10:
            X = sm.add_constant(s_lag.values)
            y = s_delta.values
            model = sm.OLS(y, X).fit()
            slope = model.params[1]
            if slope < 0:
                half_life = -np.log(2) / slope
            else:
                half_life = np.inf
        else:
            half_life = np.nan
    except:
        half_life = np.nan

    # 2. Hurst Exponent
    # Simplified R/S analysis or lag variance
    # We will use a standard R/S implementation for 'institutional' feel
    try:
        # Create a range of lags
        lags = range(2, min(len(spread)//2, 100))
        tau = []
        rs = []
        vals = spread.values
        for lag in lags:
            # Divide into chunks of size 'lag'
            # But simple Hurst is often calculated scaling the entire series
            # Let's use the 'RS' method on sliding windows? 
            # Simplified: regressing log(R/S) vs log(n)
            # R = max(Y) - min(Y), S = std(Y) of increments? 
            # Classic R/S:
            # For each chunk size n, split series, calc R/S, avg R/S.
            # Then slope of log(R/S) vs log(n) is Hurst.
            
            # Efficient check:
            # We'll stick to a faster "Aggregated Series" approximation if len is large
            # Or just calc standard R/S for a few lag points
            pass
        
        # Actually, let's substitute a robust, simple Hurst calculation:
        # log(std(diff(series) at lag tau)) vs log(tau) ? No that's for random walks.
        # Let's use the 'fluctuation analysis' approach
        ts = spread.values
        lags_h = range(2, max(3, min(len(ts)//4, 20))) 
        # Calculate variances of differences
        tau_val = []
        variances = []
        for lag in lags_h:
            # Var(y(t+tau) - y(t)) ~ tau^(2H)
            diffs = ts[lag:] - ts[:-lag]
            tau_val.append(lag)
            variances.append(np.std(diffs))
        
        # log(std) ~ H * log(tau) (approx, valid for GBM/fBm)
        # For mean reverting, this might saturate? 
        # Standard approach for trading: 
        # slope of log(Std(lag)) vs log(lag) approaches H.
        if len(variances) > 2:
            import numpy.polynomial.polynomial as poly
            log_tau = np.log(tau_val)
            log_std = np.log(variances)
            # fit line
            coefs = poly.polyfit(log_tau, log_std, 1)
            hurst = coefs[1]
        else:
            hurst = 0.5
    except:
        hurst = 0.5 

    # 3. Sharpe Ratio
    # Annualized Sharpe of the *spread returns* (not strategy PnL, but spread stability)
    # Actually, Sharpe of 'Spread PnL' if we held it? 
    # Let's just do Mean/Std of spread *changes* (since we trade mean reversion of spread levels)
    # If spread mean reverts, the 'returns' of holding the spread are the changes?
    # Simple metric: Ratio of Mean(SpreadLevel) / Std(SpreadLevel) is Z-score.
    # We want RISK adjusted return. Let's compute Sharpe of Strategy PnL? 
    # That requires backtest. 
    # Let's compute Sharpe of the Spread's "Returns" (pct_change)
    try:
        # Avoid div by zero
        # Note: Spread can be 0 or negative, so pct_change is dangerous.
        # Use diff instead.
        diffs = spread.diff().dropna()
        if diffs.std() > 1e-9:
            # Annualization factor
            factor = 1.0
            if timeframe == "1s": factor = np.sqrt(31536000)
            elif timeframe == "1min": factor = np.sqrt(525600)
            
            sharpe = (diffs.mean() / diffs.std()) * factor
        else:
            sharpe = 0.0
    except:
        sharpe = 0.0

    return {
        "Half-Life (Bars)": half_life,
        "Hurst": hurst,
        "Sharpe (Theoretical)": sharpe
    }
