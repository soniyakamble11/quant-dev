# Dockerfile - Streamlit app + ingestion worker
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install minimal system deps used by pandas/plotly/websockets etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc ffmpeg libsndfile1 curl \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose port (Render sets PORT env var). Streamlit reads STREAMLIT_SERVER_PORT.
ENV PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=${PORT}
ENV PYTHONPATH=/app

# Default command: run orchestrator (starts worker + streamlit)
CMD ["python", "run.py"]
