# ingestion_worker.py - simple worker that starts collector and keeps running
import os
import time
import signal
import sys
from ingestion import start_collector, stop_collector

def handle_sigterm(signum, frame):
    try:
        stop_collector()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

if __name__ == "__main__":
    syms = os.environ.get("SYMS", "btcusdt,ethusdt")
    print(f"[worker] starting collector for: {syms}")
    start_collector(syms)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[worker] stopping collector")
        stop_collector()
