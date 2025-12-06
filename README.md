#  Quant Research Analytics Dashboard

A fully functional real-time quantitative analytics application for ingesting market tick data, resampling OHLC bars, computing statistical arbitrage metrics, and visualizing live analytics.

This project satisfies the Quant Developer Evaluation Assignment requirements end-to-end, including:
1 WebSocket ingestion
2 Sampling & OHLC bar creation
3 Hedge ratio via OLS
4 Spread & Z-Score
5 Rolling correlation
6 ADF test
7 Alerts
8 Interactive UI
9 Data export + upload
10 Extensible modular architecture


#  Features Overview

##  **Real-Time Data Ingestion**

* Connects to Binance Futures WebSocket stream.
* Multi-symbol support (`btcusdt, ethusdt`, etc.).
* Threaded asyncio loop for stable streaming.
* Bounded in-memory buffer of ticks.
* Safe start/stop controls.

##  **Sampling & Storage**

* 1s / 1m / 5m resampled OHLCV bars.
* Robust timestamp handling (mixed formats, CSV upload).
* No dummy data — works entirely from live ticks or uploaded CSV.

##  **Analytics**

### Statistical Arbitrage Metrics (Required)

* Hedge ratio β via OLS regression.
* Spread computation.
* Rolling Z-Score.
* Rolling correlation.
* Stationarity test using Augmented Dickey-Fuller (ADF).
* Alerts when Z-Score exceeds threshold.

### Additional Analytics (Extensible)

* Framework easily supports:

  * Kalman filter hedge ratio
  * Huber / Theil‐Sen regression
  * Mean-reversion backtesting
  * Liquidity filters
  * Cross-correlation heatmaps

##  **Interactive Frontend (Streamlit)**

* Tabbed layout: **Live Data**, **Charts**, **Analytics**, **Upload**.
* Candlestick charts with zoom/pan/hover.
* Spread & Z-Score plots.
* Rolling correlation plots.
* Real-time metric updates and alerts.
* CSV upload & analysis (offline mode).
* CSV export for ticks.

##  **Alerting**

* User-defined z-score threshold.
* Alerts show in Analytics tab:

  *  Extreme Z-Score
  * High Z-Score
  *  Normal

---

#  Repository Structure

quant-antigravity/
│
├── app.py                # Main Streamlit application UI
├── ingestion.py          # WebSocket ingestion, buffer, start/stop
├── analytics.py          # Resampling + analytics + ADF + correlations
├── utils.py              # CSV export, helper functions
│
├── requirements.txt      # Dependencies
├── README.md             # This file
│
├── architecture.drawio   # Architecture diagram (source)
└── architecture.png      # Exported architecture diagram

---

#  Installation & Running the App

## 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd quant-antigravity
```

## 2️⃣ Create & activate a virtual environment

### **Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### **Mac/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3️⃣ Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4️⃣ Run the app

```bash
streamlit run app.py
```

The dashboard opens at:

```
http://localhost:8501
```

---

#  Usage Guide

##  Start Live Stream

* Click **Start** to begin collecting real-time ticks.
* Make sure symbols are listed in lowercase (e.g., `btcusdt, ethusdt`).

##  Stop Stream

* Gracefully stops the ingestion thread.

##  Clear Buffer

* Stops collection and resets tick buffer.

##  Upload CSV

Upload any OHLC CSV with columns:

```
timestamp, symbol, close
```

App automatically infers and computes analytics offline.

##  Export Ticks

* Tick-level data (live stream) can be exported as CSV.

---

#  Analytics Explanation (Required by Assignment)

## 1. **Resampling (1s / 1m / 5m)**

Ticks → OHLCV bars per symbol.
Used for regression, spread, z-score, correlation.

## 2. **Hedge Ratio (OLS Regression)**

Regression:

```
A = βB + intercept
```

β = hedge ratio for mean-reversion pair.

## 3. **Spread**

```
spread = price_A – β * price_B
```

## 4. **Rolling Z-Score**

```
z = (spread – mean) / std
```

Used to detect mean-reversion opportunities.

## 5. **ADF Test (Stationarity)**

Augmented Dickey-Fuller checks if the spread is stationary.

* p < 0.05 → stationary → good for pairs trading
* p ≥ 0.05 → non-stationary

## 6. **Rolling Correlation**

Measures strength of linear relationship between A & B.

---

#  Architecture (Required)

### Components:

* **WebSocket Ingestion Layer**

  * Async consumer per symbol
  * Buffer with thread safety
* **Tick Storage Layer**

  * In-memory buffer (extendable to Redis/DB)
* **Resampler & Analytics Engine**

  * OLS regression
  * Spread
  * Z-Score
  * ADF test
  * Rolling correlation
* **Frontend UI (Streamlit)**

  * Tabs, controls, charts, alerts
  * CSV upload/download
* **Alerting Engine**

  * Z-Score thresholds
  * Visual signals
* **Extensibility Points**

  * Alternative regressions
  * Kalman filters
  * Backtesting module

The diagram is in:

```
architecture.drawio
architecture.png
```

---

#  Design Choices & Trade-Offs (Required)

###  In-Memory Buffer

Fast for real-time analytics.
Trade-off: not persistent — but app supports saving to CSV.

###  Threaded WebSocket Collector

Avoids blocking Streamlit UI.
Trade-off: requires careful stop/kill logic.

###  Pandas for Resampling

Easy to maintain.
Trade-off: slower for >10M ticks (real deployment would use ClickHouse/TimescaleDB).

###  Streamlit Frontend

Fast prototyping, interactive widgets, Plotly support.
Trade-off: not ideal for ultra-low-latency HFT UIs.

###  Modular Code (ingestion / analytics / UI)

Simplifies future upgrades (Kalman filter, backtests, more regressions).

---

#  Extensibility Plan

Future enhancements can plug in cleanly:

* **DB Storage:** move buffer to Redis/Postgres/ClickHouse.
* **Kalman Filter Beta:** dynamic hedge estimation.
* **Backtesting Engine:** mean reversion entry/exit rules.
* **Order Book Data:** depth analytics & microstructure signals.
* **Heatmaps:** cross-correlation across multiple symbols.

---

#  Limitations (Transparency)

* In-memory only; buffer clears on restart.
* ADF test requires sufficient data (~20+ samples).
* Dependent on Binance futures API uptime.

---







---

#  AI Usage Transparency (Required)

This project was developed with the assistance of LLMs (Google Gemini / OpenAI GPT models) for:

*   **Boilerplate Code**: Streamlit UI layout and basic Plotly configurations.
*   **Mathematical Logic**: Verifying Vectorized implementations of Kalman Filters and OLS.
*   **Debugging**: Rapidly identifying `IndentationError` and `datetime` parsing issues.
*   **Architecture Design**: Refining the separation of concerns between `ingestion.py` and `analytics.py`.

**Prompts Used (Examples):**
*   "Implement a robust Kalman Filter class in pure NumPy for dynamic hedge ratio."
*   "Create a real-time Streamlit dashboard structure with tabbed layouts."
*   "Fix this specific Pandas to_datetime error for mixed formats."
