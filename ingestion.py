import asyncio
import threading
import json
import time
from datetime import datetime, timezone

import pandas as pd
import websockets

_buffer = []
_buffer_lock = threading.Lock()
_running = False
_collector_thread = None

# New globals to control asyncio loop and tasks
_loop = None
_tasks = []

def _now_iso(ts_ms=None):
    if ts_ms is None:
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).isoformat()


async def _collect_symbol_once(symbol: str):
    """
    Connect once to Binance futures trade stream for a symbol and read messages.
    Returns on normal exit or raises on error.
    """
    uri = f"wss://fstream.binance.com/ws/{symbol}@trade"
    print(f"[collector] connecting to {uri}")
    async with websockets.connect(uri) as ws:
        print(f"[collector] connected for {symbol}")
        async for msg in ws:
            # If running flag is cleared, break to allow quick shutdown
            if not _running:
                print(f"[collector] _running became False, closing connection for {symbol}")
                break

            data = json.loads(msg)
            ts = data.get("E", int(time.time() * 1000))
            price = float(data.get("p", 0.0))
            qty = float(data.get("q", 0.0))

            tick = {
                "timestamp": _now_iso(ts),
                "symbol": symbol.upper(),
                "price": price,
                "size": qty,
            }

            # append under lock
            with _buffer_lock:
                _buffer.append(tick)
                # keep buffer bounded
                if len(_buffer) > 20000:
                    del _buffer[0]

            # lightweight periodic log
            if len(_buffer) % 500 == 0:
                print(f"[collector] buffer size: {len(_buffer)}")


async def _collect_symbol_loop(symbol: str):
    """
    Continuously try to connect to the websocket for `symbol`.
    On error, wait a moment and reconnect (simple backoff).
    """
    backoff = 1.0
    while _running:
        try:
            await _collect_symbol_once(symbol)
            backoff = 1.0
        except asyncio.CancelledError:
            print(f"[collector] task cancelled for {symbol}")
            raise
        except Exception as e:
            print(f"[collector] ERROR for {symbol}: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 10.0)
    print(f"[collector] exiting loop for {symbol}")


def _run_loop(symbols: str):
    """
    Start an asyncio event loop and run a collection task per symbol.
    This runs inside the background thread.
    """
    global _loop, _tasks
    # create and set a new event loop for this thread
    loop = asyncio.new_event_loop()
    _loop = loop
    asyncio.set_event_loop(loop)

    syms = [s.strip().lower() for s in symbols.split(",") if s.strip()]
    _tasks = [loop.create_task(_collect_symbol_loop(s)) for s in syms]

    print(f"[collector] running event loop with tasks for: {syms}")
    try:
        loop.run_until_complete(asyncio.gather(*_tasks))
    except Exception as e:
        # if tasks are cancelled, this may raise; log and continue to shutdown
        print(f"[collector] loop exception: {e}")
    finally:
        # ensure all tasks are cancelled
        for t in _tasks:
            if not t.done():
                try:
                    t.cancel()
                except Exception:
                    pass
        try:
            loop.run_until_complete(asyncio.sleep(0))  # allow cancellations to propagate
        except Exception:
            pass
        loop.close()
        _loop = None
        _tasks = []
        print("[collector] event loop closed")


def start_collector(symbols: str = "btcusdt,ethusdt"):
    """
    Starts the collector in a background thread.
    """
    global _running, _collector_thread
    if _running:
        print("[collector] already running")
        return

    _running = True
    print(f"[collector] starting for symbols: {symbols}")

    _collector_thread = threading.Thread(
        target=_run_loop,
        args=(symbols,),
        daemon=True
    )
    _collector_thread.start()


def stop_collector():
    """
    Stop the collector: set running flag False, cancel asyncio tasks, and stop the loop.
    This requests cancellation from the main thread into the background loop.
    """
    global _running, _loop, _tasks
    if not _running:
        print("[collector] not running")
        return

    _running = False
    print("[collector] stop signal sent; attempting to cancel tasks and stop loop...")

    # If loop is present, ask the loop to cancel tasks and stop
    if _loop is not None:
        try:
            def _cancel_all():
                for t in list(_tasks):
                    try:
                        t.cancel()
                    except Exception:
                        pass
            # schedule cancellation in the event loop thread
            _loop.call_soon_threadsafe(_cancel_all)
        except Exception as e:
            print(f"[collector] error requesting task cancellations: {e}")

    # Optionally, wait a short time to allow tasks to cancel gracefully
    # (do not block too long in main thread)
    time.sleep(0.2)
    print("[collector] stop_collector finished request")


def clear_buffer():
    with _buffer_lock:
        _buffer.clear()
    print("[collector] buffer cleared")


def get_buffer_snapshot() -> pd.DataFrame:
    """
    Return a snapshot copy of the buffer as a pandas DataFrame.
    Uses a lock to avoid concurrent modification issues.
    """
    with _buffer_lock:
        if not _buffer:
            return pd.DataFrame(columns=["timestamp", "symbol", "price", "size"])
        data_copy = list(_buffer)
    df = pd.DataFrame(data_copy)
    return df
