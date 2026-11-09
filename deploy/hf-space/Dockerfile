# Hosted demo: one container serving the FastAPI API and the built React UI on port 7860.
FROM node:22-alpine@sha256:fe4f6d74e20370acb04ad889077d14d0f64a9c8ea6a42bc60cddf1e844699914 AS web
WORKDIR /app
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web ./
# Same-origin API: the UI calls /predict etc. on the Space's own host.
ENV VITE_API_URL=/
RUN npm run build

FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEXVIA_CHECKPOINT=/app/artifacts/best_densenet121.pt \
    NEXVIA_SEGMENTATION_CHECKPOINT=/app/artifacts/best_spleen_unet_20261109.pt \
    NEXVIA_AUDIT_PATH=/app/demo/review_events.jsonl \
    NEXVIA_ORTHANC_RESULTS_PATH=/app/demo/orthanc_predictions.jsonl \
    NEXVIA_ORTHANC_SYNC_STATE_PATH=/app/demo/orthanc_sync_state.json \
    NEXVIA_ORTHANC_SEGMENTATION_RESULTS_PATH=/app/demo/orthanc_segmentations.jsonl \
    NEXVIA_ORTHANC_SEGMENTATION_MASK_DIR=/app/demo/orthanc_masks \
    ORTHANC_AUTO_SYNC_SECONDS=0 \
    NEXVIA_DISABLE_ORTHANC=true \
    NEXVIA_RATE_LIMIT_PER_MINUTE=30 \
    NEXVIA_QA_LIMIT_PER_HOUR=20 \
    NEXVIA_TRUST_PROXY=true \
    NEXVIA_WEB_DIST=/app/web_dist

WORKDIR /app
COPY requirements-api.lock.txt requirements-torch-cpu.lock.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip==26.2.2 \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements-torch-cpu.lock.txt \
    && python -m pip install --no-cache-dir -r requirements-api.lock.txt

COPY api ./api
COPY segment_spleen.py train_segmentation.py ./
COPY artifacts ./artifacts
COPY --from=web /app/dist ./web_dist
# Hugging Face Spaces runs containers as UID 1000. Keep only the demo state folder writable.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/demo \
    && chown -R app:app /app/demo
USER app
EXPOSE 7860
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
