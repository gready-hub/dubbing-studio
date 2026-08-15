FROM python:3.12-slim

# ffmpeg does the media work; the rest keeps the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DUBBING_STUDIO_DOCKER=1 \
    DUBBING_STUDIO_HOME=/data \
    DUBBING_STUDIO_OUTPUT=/output

WORKDIR /srv

COPY requirements-base.txt requirements-portable.txt ./
RUN pip install --no-cache-dir -r requirements-portable.txt

COPY app ./app

RUN mkdir -p /data /output
VOLUME ["/data", "/output"]
EXPOSE 8765

# Models land in /data, so they survive container restarts.
CMD ["python", "-m", "app.server", "8765", "--no-browser"]
