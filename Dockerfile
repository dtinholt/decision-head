# decisionhead has zero runtime dependencies: this image needs nothing
# beyond the standard library, so there is no build stage and no compiler.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="decisionhead" \
      org.opencontainers.image.description="Self-hosted typed-question decision service" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/dtinholt/decision-head"

WORKDIR /app

# Only the files needed to install the package -- no test suite, no docs,
# no dev tooling -- keeps the image small and the build cache stable.
COPY pyproject.toml README.md LICENSE ./
COPY decisionhead ./decisionhead

RUN pip install --no-cache-dir --no-deps .

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin decisionhead
USER decisionhead

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["decisionhead", "serve"]
# No arguments by default, so the server binds 127.0.0.1 (the config
# default) and every DECISIONHEAD_* environment variable still applies
# (a CLI flag here would override them). To publish the port, set
# DECISIONHEAD_HOST=0.0.0.0, DECISIONHEAD_ALLOW_NON_LOCAL_BIND=true and
# DECISIONHEAD_BEARER_TOKEN_ENV (plus that variable): a non-loopback bind
# without a bearer token is refused (2026-10-06 security review, finding 7).
CMD []
