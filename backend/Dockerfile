FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VECTOR_STORE_PATH=/data/vector_store \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /srv/backend
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home app \
    && mkdir -p /data/vector_store \
    && chown -R app:app /data /srv/backend
COPY --chown=app:app app ./app
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
