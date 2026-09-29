FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends bash ripgrep procps && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir charset-normalizer==3.4.7
RUN mkdir -p /filesystem /app/.feather && chmod 777 /filesystem /app/.feather
CMD ["sleep", "infinity"]
