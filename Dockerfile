# syntax=docker/dockerfile:1
# Sieve server image: slim, non-root, no build tools in the final layer.
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build \
 && python -m build --wheel --outdir /wheels .

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY --from=build /wheels /wheels
RUN pip install "$(ls /wheels/*.whl)[server]" && rm -rf /wheels \
 && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin sieve \
 && mkdir -p /etc/sieve /var/lib/sieve && chown sieve /var/lib/sieve
USER 10001
WORKDIR /var/lib/sieve
EXPOSE 8080
# /healthz is unauthenticated liveness; use /readyz for orchestrator readiness gating.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"]
ENTRYPOINT ["sieve"]
CMD ["serve", "--config", "/etc/sieve/config.yaml"]
