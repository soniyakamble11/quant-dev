# System Architecture

This document describes the high-level architecture of the Quant Research Dashboard.

```mermaid
graph TD
    subgraph "Data Sources"
        WS[Binance WebSocket] -->|JSON Ticks| Ingestion
        CSV[CSV Upload] -->|Pandas DF| Ingestion
    end

    subgraph "Ingestion Layer (ingestion.py)"
        Ingestion[Async Collector]
        Buffer[In-Memory Tick Buffer]
        Ingestion --> Buffer
    end

    subgraph "Processing & Analytics (analytics.py)"
        Resampler[Resampler (1s/1m/5m)]
        Buffer --> Resampler
        
        subgraph "Stat Arb Engine"
            OLS[OLS Regression]
            Kalman[Kalman Filter (Dynamic Beta)]
            ADF[ADF Stationarity Test]
            Spread[Spread & Z-Score Calc]
        end
        
        subgraph "Risk & Alpha Metrics"
            Micro[Microstructure (Vol, Arrival)]
            Risk[Risk Metrics (Sharpe, Hurst, Half-Life)]
            Corr[Correlation Matrix]
        end
        
        subgraph "Strategy"
            Backtest[Mean Reversion Backtester]
        end
        
        Resampler --> OLS
        Resampler --> Kalman
        Resampler --> Micro
        Resampler --> Corr
        OLS --> Spread
        Kalman --> Spread
        Spread --> ADF
        Spread --> Risk
        Spread --> Backtest
        
    end

    subgraph "Frontend UI (app.py)"
        Session[Session State (Persistence)]
        
        Tabs[Streamlit Tabs]
        Live[Live Data Tab]
        Charts[Charts Tab]
        Analytics[Analytics Tab]
        StrategyTab[Strategy Monitor]
        
        Tabs --> Live
        Tabs --> Charts
        Tabs --> Analytics
        Tabs --> StrategyTab
        
        Live -->|Display| Micro
        Charts -->|Plot| Spread
        Charts -->|Plot| Corr
        Analytics -->|Display| Risk
        Analytics -->|Display| Kalman
        StrategyTab -->|Run| Backtest
    end

    style WS fill:#f9f,stroke:#333,stroke-width:2px
    style Kalman fill:#bbf,stroke:#333,stroke-width:2px
    style Backtest fill:#bfb,stroke:#333,stroke-width:2px
    style Risk fill:#fbf,stroke:#333,stroke-width:2px
```

## Component Description

1.  **Ingestion Layer**: Handles real-time connections to Binance via `websockets`. Buffers ticks in a thread-safe deque.
2.  **Processing Engine**:
    *   **Resampler**: Converts raw ticks to OHLCV bars.
    *   **Stat Arb Engine**: Computes hedge ratios using either static OLS or dynamic **Kalman Filter**.
    *   **Risk Metrics**: Calculates institutional metrics like **Hurst Exponent** and **Half-Life** to gauge signal quality.
3.  **UI Layer**: Built with Streamlit for rapid interactive visualization. Uses `st.session_state` to persist backtest results and uploads across re-runs.
