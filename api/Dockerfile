FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEXVIA_CHECKPOINT=/app/artifacts/best_densenet121.pt \
    NEXVIA_AUDIT_PATH=/app/demo/review_events.jsonl

WORKDIR /app
COPY requirements-api.lock.txt requirements-torch-cpu.lock.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip==26.2.2 \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements-torch-cpu.lock.txt \
    && python -m pip install --no-cache-dir -r requirements-api.lock.txt

COPY api ./api
COPY segment_spleen.py train_segmentation.py ./
# Run as an unprivileged user. UID 1000 matches the default user on most Linux hosts, so the
# bind-mounted ./demo folder stays writable.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/demo \
    && chown -R app:app /app/demo
USER app
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
