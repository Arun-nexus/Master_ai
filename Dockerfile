FROM node:22-slim AS web
WORKDIR /web
COPY ui/package*.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    MCP_SERVER=erp_backend_tools.py

WORKDIR /app

COPY requirements.txt .
RUN grep -viE '^(pywin32|torch|torchaudio|torchvision)==' requirements.txt > req.txt \
 && pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r req.txt

COPY api.py erp_backend_tools.py configuration.py data_extraction.py \
     logger.py model_call.py schemas.py vector_store.py params.yaml .env* ./
COPY --from=web /web/dist ./static

VOLUME ["/app/Data", "/app/logs", "/app/.hf_cache"]

EXPOSE 8765

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8765"]