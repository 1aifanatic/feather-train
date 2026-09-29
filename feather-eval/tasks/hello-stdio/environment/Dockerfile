FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && pip3 install --break-system-packages 'mcp==1.30.0' \
    && rm -rf /var/lib/apt/lists/*
COPY server.py /usr/local/bin/hello-stdio-server
RUN chmod 755 /usr/local/bin/hello-stdio-server \
    && useradd --create-home agent \
    && mkdir -p /app && chown agent:agent /app
WORKDIR /app
