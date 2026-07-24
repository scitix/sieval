FROM python:3.10-slim

# nodejs
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -sL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# python evaluator
WORKDIR /opt/code-evaluator

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["fastapi", "run", "app/server.py", "--port", "11451"]
