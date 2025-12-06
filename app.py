
import asyncio
# Fix Windows event-loop issues for websockets
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass


import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import logging
from datetime import datetime, timezone
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# ---------------- Auto-refresh (2s) ----------------
_HAS_AUTOREF = False
try:
    from streamlit_autorefresh import st_autorefresh
    # 2000 ms = 2 seconds
    st_autorefresh(interval=2000, key="auto_refresh")
    _HAS_AUTOREF = True
except Exception:
    # streamlit-autorefresh not installed; fallback: no automatic refresh
    _HAS_AUTOREF = False

# ---------- helpers ----------
def safe_last(series, default=None):
    try:
        if series is None:
            return default
        s = series.dropna()
        if s.empty:
            return default
        return float(s.iloc[-1])
    except Exception:
        return default

def csv_with_meta(df: pd.DataFrame, meta: dict) -> bytes:
    """
    Produce CSV bytes with a short metadata header (lines starting with #).
    """
    meta_lines = [f"# {k}: {v}" for k, v in meta.items()]
    meta_block = "\n".join(meta_lines) + "\n"
    csv_bytes = df_to_csv(df)
    if isinstance(csv_bytes, (bytes, bytearray)):
        return meta_block.encode() + csv_bytes
    else:
        return meta_block.encode() + str(csv_bytes).encode()

# ---------- logging ----------
if not os.path.exists("logs"):
    try:
        os.makedirs("logs")
    except Exception:
        pass
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', filename="logs/app.log")
logger = logging.getLogger(__name__)

# ---------- imports from your modules ----------
from ingestion import start_collector, stop_collector, get_buffer_snapshot, clear_buffer
from analytics import compute_ols_hedge, compute_spread_zscore, run_adf, resample_bars, compute_rolling_correlation, run_backtest, compute_kalman_hedge, compute_microstructure_metrics, compute_correlation_matrix, compute_volatility, compute_risk_metrics
from utils import df_to_csv

# -------------------- Page Setup ----------------------
st.set_page_config(page_title="Quant Research Dashboard", layout="wide")

# Title
st.markdown("""
    <h1 style="text-align:center; color:#0ea5a4; margin-bottom:2px;">
        Quant Research Analytics Dashboard
    </h1>
    <p style="text-align:center; font-size:14px; color:#6b7280; margin-top:0px;">
        Live Market Data • Statistical Arbitrage • Real-Time Visualization
    </p>
""", unsafe_allow_html=True)

# ---------------- Session defaults ----------------
if "export_bytes" not in st.session_state:
    st.session_state["export_bytes"] = None
if "upload_df" not in st.session_state:
    st.session_state["upload_df"] = None
if "recent_z_alerts" not in st.session_state:
    st.session_state["recent_z_alerts"] = []

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Controls & Alerts")

# Instrument / sampling
st.sidebar.subheader("Instrument Setup")
symbols_input = st.sidebar.text_input("Symbols ", value="btcusdt,ethusdt", key="symbols_input")
timeframe = st.sidebar.selectbox("Sampling Interval", ["1s", "1min", "5min"], index=0, key="timeframe")
rolling_window = st.sidebar.slider("Rolling Window (bars)", 10, 300, 60, key="rolling_window")

# Regression control
reg_type = st.sidebar.selectbox("Regression Type", ["OLS", "Robust (Huber)", "Kalman Filter"], index=0, key="reg_type")

# Alerts editor
st.sidebar.subheader("🚨 Alert Rules")
alert_metric = st.sidebar.selectbox("Metric", ["Z-Score", "Spread", "Rolling Correlation"], index=0, key="alert_metric")
alert_op = st.sidebar.selectbox("Operator", [">", "<"], index=0, key="alert_op")
alert_threshold = st.sidebar.number_input("Threshold", value=2.0, step=0.1, key="alert_threshold")
alert_enabled = st.sidebar.checkbox("Enable Alerting", value=True, key="alert_enabled")

# Demo mode (lowers thresholds)
demo_mode = st.sidebar.checkbox("Demo Mode (lower thresholds)", value=False, key="demo_mode")
if demo_mode:
    # make threshold smaller for demo
    alert_threshold = min(alert_threshold, 0.5)

# Upload
st.sidebar.subheader("📤 Upload Historical CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV (timestamp, symbol, close)", type=["csv"], key="file_uploader")

# ---------------- Top actions ----------------
colA, colB, colC, colD, colE = st.columns([1,1,1,1,1])
with colA:
    if st.button("▶️ Start Stream", key="btn_start"):
        syms = ",".join([s.strip().lower() for s in symbols_input.split(",") if s.strip()])
        start_collector(syms)
        st.success("Started collecting real-time data")
        logger.info(f"Collector started: {syms}")
with colB:
    if st.button("⏹ Stop Stream", key="btn_stop"):
        stop_collector()
        st.info("Stopped collection")
        logger.info("Collector stopped")
with colC:
    if st.button("🧹 Clear Buffer", key="btn_clear"):
        stop_collector()
        clear_buffer()
        st.warning("Buffer cleared")
        logger.info("Buffer cleared")
with colD:
    if st.button("🔁 Refresh View", key="btn_refresh"):
        # manual refresh button; st_autorefresh will also update automatically every 2s
        try:
            # try the canonical API
            st.experimental_rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                # no-op if running environment doesn't support programmatic rerun
                pass
with colE:
    if st.button("💾 Prepare Export (ticks)", key="btn_prepare_export"):
        df_ticks = get_buffer_snapshot()
        if df_ticks.empty:
            st.error("No ticks to export")
        else:
            csv_bytes = df_to_csv(df_ticks)
            if csv_bytes:
                st.session_state["export_bytes"] = csv_bytes
                st.success(f"Prepared CSV ({len(csv_bytes)} bytes). Use download button.")
            else:
                st.error("CSV export failed. See logs.")
                logger.warning("CSV export failed - empty bytes")

if st.session_state["export_bytes"]:
    st.download_button("Download ticks CSV", st.session_state["export_bytes"], file_name=f"ticks_export_{int(time.time())}.csv", mime="text/csv", key="btn_download_ticks")

st.markdown("---")

# ---------------- Tabs ----------------
tab_live, tab_charts, tab_analytics, tab_strategy, tab_upload = st.tabs(["📡 Live Data", "📈 Charts", "📊 Analytics", "🛠 Strategy Monitor", "📤 Upload"])

# ---------------- Tab: Live Data ----------------
with tab_live:
    st.header("📡 Live Market Data")
    df_ticks = get_buffer_snapshot()
    st.metric("Buffered Ticks", len(df_ticks))
    if not df_ticks.empty:
        st.dataframe(df_ticks.tail(20), height=300)
        
        st.markdown("---")
        st.subheader("🛡️ Market Microstructure (Last 60s)")
        
        # Microstructure metrics
        micro_df = compute_microstructure_metrics(df_ticks)
        if not micro_df.empty:
            st.dataframe(micro_df.style.format({
                "Arrival Rate (Hz)": "{:.2f}",
                "Trade Vol (Qty)": "{:.4f}",
                "Quote Vol ($)": "{:,.2f}"
            }))
        else:
            st.info("Insufficient data for microstructure metrics.")
    else:
        st.info("Waiting for live tick data... Press Start")

# ---------------- Handle upload parsing ----------------
if uploaded_file is not None:
    try:
        df_up = pd.read_csv(uploaded_file)
        if 'timestamp' in df_up.columns:
            df_up["timestamp"] = pd.to_datetime(df_up["timestamp"], utc=True, errors="coerce")
        else:
            for c in ['time','date','datetime']:
                if c in df_up.columns:
                    df_up["timestamp"] = pd.to_datetime(df_up[c], utc=True, errors="coerce")
                    break
        df_up = df_up.dropna(subset=["timestamp"])
        if "price" not in df_up.columns and "close" in df_up.columns:
            df_up["price"] = df_up["close"]
        st.session_state["upload_df"] = df_up
        st.success("Uploaded CSV parsed")
        logger.info("CSV uploaded and parsed")
    except Exception as e:
        st.sidebar.error(f"Upload parse error: {e}")
        logger.exception("Upload parse error")

# ---------------- Choose data source ----------------
if "upload_df" in st.session_state and st.session_state["upload_df"] is not None:
    source_choice = st.radio("Data Source:", ["Live Stream", "Uploaded CSV"], index=0, key="data_source")
else:
    source_choice = "Live Stream"

df_ticks = get_buffer_snapshot()
df_source = df_ticks.copy() if source_choice == "Live Stream" else st.session_state.get("upload_df", pd.DataFrame())

# ---------------- Parse timestamps & resample bars ----------------
if df_source is None or df_source.empty:
    bars = pd.DataFrame()
else:
    # defensive copy
    df_source = df_source.copy()
    df_source["timestamp"] = pd.to_datetime(df_source["timestamp"], utc=True, errors="coerce")
    df_source = df_source.dropna(subset=["timestamp"])
    if df_source.empty:
        bars = pd.DataFrame()
    else:
        df_source = df_source.sort_values("timestamp").set_index("timestamp")
        bars = resample_bars(df_source, rule=timeframe)

if bars is None:
    bars = pd.DataFrame()
symbols = sorted(list(bars["symbol"].unique())) if (not bars.empty and "symbol" in bars.columns) else []

# ---------------- Tab: Charts ----------------
with tab_charts:
    st.header("📈 Market Charts & Exports")

    # Candlestick
    st.subheader("Candlestick Chart")
    candle_symbol = st.selectbox("Symbol (Candles)", options=symbols if symbols else ["-"], index=0, key="candle_symbol")
    if symbols and candle_symbol != "-":
        df_candle = bars[bars["symbol"] == candle_symbol]
        if not df_candle.empty:
            fig_candle = go.Figure(data=[go.Candlestick(x=df_candle.index,
                                                        open=df_candle["open"],
                                                        high=df_candle["high"],
                                                        low=df_candle["low"],
                                                        close=df_candle["close"])])
            fig_candle.update_layout(title=f"{candle_symbol.upper()} Candlestick ({timeframe})",
                                     height=480, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig_candle, use_container_width=True)

            meta = {"symbol": candle_symbol, "timeframe": timeframe, "exported_at": datetime.now(timezone.utc).isoformat()}
            csv_bars = csv_with_meta(df_candle.reset_index(), meta)
            st.download_button("Export Bars CSV", csv_bars, file_name=f"{candle_symbol}_bars_{timeframe}.csv", mime="text/csv")
        else:
            st.info("Not enough bars to render candlestick yet.")
    else:
        st.info("No symbols available for candlestick. Collect data or upload CSV.")

    st.markdown("---")

    # Rolling Volatility
    st.subheader("Rolling Volatility (Annualized)")
    if symbols:
        vol_df = compute_volatility(bars, window=rolling_window, timeframe=timeframe)
        if not vol_df.empty:
            # show multi-line chart for all symbols or just one?
            # All symbols on one chart is nice for comparison
            fig_vol = px.line(vol_df, x=vol_df.index, y="volatility", color="symbol", title="Annualized Volatility")
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Export
            meta_vol = {"timeframe": timeframe, "generated_at": datetime.now(timezone.utc).isoformat()}
            st.download_button("Export Volatility CSV", csv_with_meta(vol_df, meta_vol), file_name=f"volatility_{timeframe}.csv", mime="text/csv")
        else:
            st.info("Insufficient data for volatility.")
    
    st.markdown("---")

    # Spread & Z-score
    st.subheader("Spread & Z-Score")
    if len(symbols) < 2:
        st.info("Need at least 2 symbols to compute pair analytics.")
        series_a = pd.Series(dtype=float)
        series_b = pd.Series(dtype=float)
        beta = intercept = r2 = 0.0
        spread = zscore = pd.Series(dtype=float)
    else:
        sym_a = st.selectbox("Symbol A", options=symbols, index=0, key="sym_a")
        sym_b = st.selectbox("Symbol B", options=symbols, index=1 if len(symbols) > 1 else 0, key="sym_b")

        series_a = bars[bars["symbol"] == sym_a]["close"] if not bars.empty else pd.Series(dtype=float)
        series_b = bars[bars["symbol"] == sym_b]["close"] if not bars.empty else pd.Series(dtype=float)

        # compute regression with spinner
        with st.spinner("Computing regression, spread and z-score..."):
            try:
                if reg_type == "OLS":
                    beta, intercept, r2 = compute_ols_hedge(series_a, series_b)
                elif reg_type == "Kalman Filter":
                    beta, intercept, series_beta, series_alpha = compute_kalman_hedge(series_a, series_b)
                    r2 = 0.999 # Pseudo R2 or just arbitrary high for Kalman as it tracks closely but varies
                else: # Robust
                    df_pair = pd.concat([series_a, series_b], axis=1).dropna()
                    if df_pair.shape[0] < 5:
                        beta, intercept, r2 = 1.0, 0.0, 0.0
                    else:
                        y = df_pair.iloc[:,0].astype(float)
                        X = df_pair.iloc[:,1].astype(float)
                        Xc = sm.add_constant(X)
                        rlm = sm.RLM(y, Xc, M=sm.robust.norms.HuberT()).fit()
                        beta = float(rlm.params[1]) if len(rlm.params) > 1 else 1.0
                        intercept = float(rlm.params[0]) if len(rlm.params) > 0 else 0.0
                        r2 = 0.0
            except Exception as e:
                logger.exception("Regression failure")
                beta, intercept, r2 = 1.0, 0.0, 0.0

            # If Kalman, spread is A - beta_t * B - alpha_t
            if reg_type == "Kalman Filter" and not series_beta.empty:
                # Re-align
                df_k = pd.concat([series_a, series_b, series_beta, series_alpha], axis=1).dropna()
                # spread = A - beta_t * B - alpha_t (innovation process approximately)
                # Note: spread_zscore function expects scalar beta. We need to handle dynamic spread manually here if selected.
                # However, for consistent z-score logic, we'll calculate dynamic spread here and then pass dummy beta=1, intercept=0 to compute_spread_zscore?
                # Actually compute_spread_zscore does logic inside. Let's manually compute spread if Kalman.
                s_a = df_k.iloc[:, 0]
                s_b = df_k.iloc[:, 1]
                b_t = df_k.iloc[:, 2]
                a_t = df_k.iloc[:, 3]
                
                spread = s_a - b_t * s_b - a_t
                
                # Manual z-score on this dynamic spread
                roll_mean = spread.rolling(rolling_window).mean()
                roll_std = spread.rolling(rolling_window).std().replace(0, np.nan)
                zscore = (spread - roll_mean) / roll_std
            else:
                spread, zscore = compute_spread_zscore(series_a, series_b, beta, rolling_window)

        # Spread plot + export
        if not spread.empty:
            fig_spread = px.line(spread, title=f"Spread: {sym_a} - β*{sym_b}")
            st.plotly_chart(fig_spread, use_container_width=True)

            df_spread = pd.concat([spread.rename("spread"), zscore.rename("zscore")], axis=1)
            meta = {"symbols": f"{sym_a},{sym_b}", "timeframe": timeframe, "regression": reg_type, "generated_at": datetime.now(timezone.utc).isoformat()}
            st.download_button("Export Spread & Z-score CSV", csv_with_meta(df_spread.reset_index(), meta), file_name=f"spread_zscore_{sym_a}_{sym_b}.csv", mime="text/csv")
        else:
            st.info("Spread not available yet (insufficient aligned bars).")

        # Z-score plot (with chart annotation possibility)
        if not zscore.empty:
            fig_z = px.line(zscore, title="Rolling Z-Score")
            # Chart annotation if alert on zscore triggers
            latest_z = safe_last(zscore, default=None)
            # We'll annotate when alert fired on analytics stage (kept below) by adding vline & annotation if needed.
            st.plotly_chart(fig_z, use_container_width=True)
        else:
            st.info("Z-score not available yet (need more bars).")
            
        st.markdown("---")

        # Rolling Correlation
        st.subheader("Rolling Correlation")
        # Compute here for Charts tab
        rolling_corr_chart = compute_rolling_correlation(series_a, series_b, rolling_window) if (not series_a.empty and not series_b.empty) else pd.Series(dtype=float)
        
        if not rolling_corr_chart.empty:
            fig_corr = px.line(rolling_corr_chart, title="Rolling Correlation")
            st.plotly_chart(fig_corr, use_container_width=True)
            # export
            meta_corr = {"symbols": f"{sym_a},{sym_b}", "timeframe": timeframe, "generated_at": datetime.now(timezone.utc).isoformat()}
            st.download_button("Export Rolling Correlation CSV", csv_with_meta(rolling_corr_chart.reset_index().rename(columns={0:"corr"}), meta_corr), file_name=f"rolling_corr_{sym_a}_{sym_b}.csv", mime="text/csv")
        else:
            st.info("Rolling correlation not available yet (need more aligned bars).")

# ---------------- Tab: Analytics ----------------
with tab_analytics:
    st.header("📊 Statistical Analytics & Alerts")

    if len(symbols) < 2:
        st.info("Need at least 2 symbols to show analytics.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hedge Ratio β", f"{beta:.4f}")
        col2.metric("Intercept", f"{intercept:.4f}")
        col3.metric("R²", f"{r2:.4f}")
        
        # Spread Volatility
        spread_vol = 0.0
        if not spread.empty:
            spread_vol = spread.rolling(rolling_window).std().iloc[-1]
            if pd.isna(spread_vol): spread_vol = 0.0
        col4.metric("Spread Vol (Window)", f"{spread_vol:.4f}")

        # Evaluate alert condition
        alert_fired = False
        alert_msg = ""
        metric_val = None

        # compute rolling correlation on demand
        rolling_corr = compute_rolling_correlation(series_a, series_b, rolling_window) if (not series_a.empty and not series_b.empty) else pd.Series(dtype=float)

        # choose the metric value
        if alert_metric == "Z-Score":
            metric_val = safe_last(zscore, default=None)
        elif alert_metric == "Spread":
            metric_val = safe_last(spread, default=None)
        elif alert_metric == "Rolling Correlation":
            metric_val = safe_last(rolling_corr, default=None)

        if alert_enabled and metric_val is not None:
            if alert_op == ">" and metric_val > alert_threshold:
                alert_fired = True
            elif alert_op == "<" and metric_val < alert_threshold:
                alert_fired = True

        # Show global sticky banner if fired (renders on top)
        if alert_fired:
            alert_msg = f"ALERT: {alert_metric} {metric_val:.4f} {alert_op} {alert_threshold}"
            st.markdown(f"""
                <div style="position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:9999;
                            padding:12px 18px;border-radius:8px;background:#fff5f5;border:1px solid #fecaca;">
                    <strong style="color:#b91c1c">🔥 {alert_msg}</strong>
                </div>
            """, unsafe_allow_html=True)
            st.sidebar.error(alert_msg)
            # also show banner in analytics tab
            st.markdown(f"""
                <div style="padding:12px;border-radius:6px;background:#ffefef;border:1px solid #ff7b7b;margin-bottom:10px;">
                    <strong style="color:#b91c1c">🔥 {alert_msg}</strong>
                </div>
            """, unsafe_allow_html=True)
        else:
            if metric_val is not None:
                st.info(f"Latest {alert_metric}: {metric_val:.4f} — no alert (threshold {alert_op} {alert_threshold})")
            else:
                st.info("Metrics not available yet for alerting.")

        # --- Recent specific Z > 2 / Z < -2 Alerts (User Request) ---
        latest_z_val_for_log = safe_last(zscore, default=None)
        if latest_z_val_for_log is not None:
            # Check for strict > 2 or < -2
            if abs(latest_z_val_for_log) > 2.0:
                last_ts = zscore.dropna().index[-1]
                condition_str = "Z > 2.0" if latest_z_val_for_log > 2.0 else "Z < -2.0"
                
                # Deduplicate based on timestamp to avoid spamming the same bar's alert
                existing_timestamps = [a["Timestamp"] for a in st.session_state["recent_z_alerts"]]
                if str(last_ts) not in existing_timestamps:
                    new_alert = {
                        "Timestamp": str(last_ts),
                        "Value": float(f"{latest_z_val_for_log:.4f}"),
                        "Condition": condition_str
                    }
                    st.session_state["recent_z_alerts"].insert(0, new_alert)
                    # Keep max 5
                    if len(st.session_state["recent_z_alerts"]) > 5:
                        st.session_state["recent_z_alerts"] = st.session_state["recent_z_alerts"][:5]
        
        st.subheader("Recent Z-Score Alerts (|Z| > 2)")
        if st.session_state["recent_z_alerts"]:
            st.table(pd.DataFrame(st.session_state["recent_z_alerts"]))
        else:
            st.info("No Z-score outliers detected in this session yet (requires |Z| > 2).")



        # ADF test (human-friendly)
        st.subheader("ADF Stationarity Test")
        if st.button("Run ADF Test", key="btn_adf"):
            adf_res = run_adf(spread.dropna()) if (not spread.empty) else {"error": "spread not available"}
            if "error" in adf_res:
                st.error(adf_res["error"])
            else:
                st.write(f"Test statistic: {adf_res['test_stat']:.4f}")
                st.write(f"p-value: {adf_res['pvalue']:.4f}")
                st.write(f"used lags: {adf_res['usedlag']}")
                st.write("Critical values:")
                st.json(adf_res.get("crit_values", {}))
                if adf_res.get("pvalue") is not None:
                    if adf_res["pvalue"] < 0.05:
                        st.success("Interpretation: Spread appears STATIONARY (p < 0.05).")
                    else:
                        st.warning("Interpretation: Spread appears NON-STATIONARY (p >= 0.05).")
        
        st.markdown("---")
        
        # Risk & Alpha stats
        st.subheader("⚖️ Risk & Alpha Metrics")
        if not spread.empty and len(spread) > 20:
            risk_metrics = compute_risk_metrics(spread, timeframe=timeframe)
            c_r1, c_r2, c_r3 = st.columns(3)
            
            hl = risk_metrics["Half-Life (Bars)"]
            hurst = risk_metrics["Hurst"]
            sharpe = risk_metrics["Sharpe (Theoretical)"]
            
            c_r1.metric("Half-Life (Bars)", f"{hl:.1f}" if not np.isnan(hl) and not np.isinf(hl) else "Inf")
            c_r2.metric("Hurst Exp", f"{hurst:.3f}", delta="Mean Reverting" if hurst < 0.5 else "Trending", delta_color="inverse")
            c_r3.metric("Sharpe (Spread)", f"{sharpe:.2f}")
        else:
            st.info("Need more data (>20 bars) for risk metrics.")

        # (Backtest moved to Strategy Monitor tab)

        # If alert fired for zscore, annotate the zscore chart (if present)
        if alert_fired and alert_metric == "Z-Score" and (not zscore.empty):
            try:
                # annotate last zscore time on z-score figure if exists on Charts tab (we recreate fig here)
                fig_z2 = px.line(zscore, title="Rolling Z-Score (annotated)")
                t_idx = zscore.dropna().index[-1]
                t_val = float(zscore.dropna().iloc[-1])
                fig_z2.add_vline(x=t_idx, line_dash="dash", line_color="red")
                fig_z2.add_annotation(x=t_idx, y=t_val, text="ALERT", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor="rgba(255,255,255,0.6)")
                st.plotly_chart(fig_z2, use_container_width=True)
            except Exception:
                logger.exception("Failed to annotate zscore chart")

# ---------------- Tab: Strategy Monitor ----------------
with tab_strategy:
    st.header("🛠 Strategy Monitor & Backtest")
    st.write("Simulate a mean-reversion strategy on the current loaded data.")
    
    st.markdown("### Parameters")
    c_bt1, c_bt2, c_bt3 = st.columns(3)
    bt_entry = c_bt1.number_input("Entry Cutoff (|Z| >)", value=2.0, step=0.1, min_value=0.5, max_value=5.0, key="bt_entry")
    bt_exit = c_bt2.number_input("Exit Cutoff (|Z| <)", value=0.0, step=0.1, min_value=-2.0, max_value=2.0, key="bt_exit")
    
    if "backtest_results" not in st.session_state:
        st.session_state["backtest_results"] = None

    if 'spread' not in locals() or 'zscore' not in locals() or spread.empty or zscore.empty:
         st.warning("Please ensure symbols are loaded and analytics computed (check Charts tab).")
    else:
        if c_bt3.button("Run Backtest", key="btn_run_bt"):
            with st.spinner("Running backtest..."):
                bt_res = run_backtest(spread, zscore, entry_cutoff=bt_entry, exit_cutoff=bt_exit)
                st.session_state["backtest_results"] = bt_res

    # Render results (same as before)
    if st.session_state["backtest_results"] is not None:
        bt_res = st.session_state["backtest_results"]
        
        if "error" in bt_res:
            st.error(bt_res["error"])
        else:
            stats = bt_res["stats"]
            trades_df = bt_res["trades"]
            equity_curve = bt_res["equity"]
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{stats['Total Return']:.4f}")
            m2.metric("Win Rate", f"{stats['Win Rate']*100:.1f}%")
            m3.metric("Max Drawdown", f"{stats['Max Drawdown']:.4f}")
            m4.metric("Trades", stats["Trade Count"])
            
            # Equity Curve
            fig_eq = px.line(equity_curve, title="Strategy Equity Curve (Spread PnL)")
            st.plotly_chart(fig_eq, use_container_width=True, key="backtest_chart_strategy")
            
            # Trades List
            st.write("Recent Trades:")
            if not trades_df.empty:
                st.dataframe(trades_df.sort_values("Entry Time", ascending=False).head(10))
                
                # Export
                csv_bt = csv_with_meta(trades_df, {"strategy": "mean_reversion", "entry": bt_entry, "exit": bt_exit})
                st.download_button("Download Full Trades CSV", csv_bt, "backtest_trades.csv", "text/csv")
            else:
                st.info("No trades generated with current parameters.")
with tab_upload:
    st.header("📤 Upload Mode")
    st.write("Upload a CSV file to run analytics without live streaming.")
    if "upload_df" in st.session_state and st.session_state["upload_df"] is not None:
        st.success("Upload received:")
        st.dataframe(st.session_state["upload_df"].head())
    else:
        st.info("No uploaded file in session.")

# ----------------- End ----------------
