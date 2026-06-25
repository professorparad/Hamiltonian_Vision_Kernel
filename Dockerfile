FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml setup.py README.md LICENSE ./
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY Main ./Main
COPY Main2 ./Main2
COPY Baselines ./Baselines
COPY tests ./tests

ENTRYPOINT ["python", "Main/main.py"]
