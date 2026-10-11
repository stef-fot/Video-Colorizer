# ColorStream: Python colorization server + Stremio addon in one image.
# GPU: run with the NVIDIA Container Toolkit (see docker-compose.yml).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-cuda.txt ./
RUN pip3 install -r requirements-cuda.txt

COPY addon/package.json addon/package-lock.json* addon/
RUN cd addon && npm install --omit=dev

COPY . .
RUN chmod +x docker/entrypoint.sh

# server api, dashboard, stremio addon
EXPOSE 7002 7001 7000
VOLUME ["/app/cache", "/app/weights"]
ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["all"]
