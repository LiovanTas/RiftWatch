# RiftWatch website. Build: docker compose build web; run: docker compose up -d
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Dependencies first, so a code change doesn't reinstall them.
COPY pyproject.toml README.md ./
RUN mkdir -p src/riftwatch && touch src/riftwatch/__init__.py \
 && pip install ".[web,ml]" && pip uninstall -y riftwatch  && rm -rf build src
COPY src ./src
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 riftwatch
USER riftwatch

ENV RIFTWATCH_MODELS_DIR=/app/models WEB_WORKERS=2
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
# Migrations are idempotent, so every start brings the schema up to date first.
CMD ["sh", "-c", "riftwatch db migrate && exec riftwatch serve --host 0.0.0.0 --port 8000 --workers ${WEB_WORKERS}"]
