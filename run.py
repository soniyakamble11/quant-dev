# run.py - orchestrator to start ingestion worker and streamlit
import subprocess
import os
import signal
import sys
import time

def start_worker():
    # Start worker in background; if you already have an ingestion entrypoint, replace the command.
    if os.path.exists("ingestion_worker.py"):
        return subprocess.Popen([sys.executable, "ingestion_worker.py"])
    # fallback: try to import ingestion.start_collector inside a subprocess
    return None

def start_streamlit():
    port = os.environ.get("PORT", "8080")
    cmd = ["streamlit", "run", "app.py", "--server.port", port, "--server.headless", "true"]
    return subprocess.Popen(cmd)

if __name__ == "__main__":
    worker = None
    try:
        worker = start_worker()
    except Exception as e:
        print("Failed to start worker:", e)

    try:
        st_proc = start_streamlit()
    except Exception as e:
        print("Failed to start streamlit:", e)
        if worker:
            worker.terminate()
        raise

    try:
        # Wait until streamlit exits; keep process alive
        st_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if worker:
            try:
                worker.terminate()
            except Exception:
                pass
        try:
            st_proc.terminate()
        except Exception:
            pass
        sys.exit(0)
