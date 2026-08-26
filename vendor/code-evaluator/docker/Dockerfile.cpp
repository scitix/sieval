# Image for the `liveoibench` source: the base image has no C++ toolchain, and
# `exec_cpp` shells out to `g++`. `libstdc++-*-dev` and `libc6-dev` are what the
# upstream judge's `-static` link needs; without them g++ is present but every
# submission fails to link, which reads as a benchmark of nothing.
FROM python:3.10-slim

WORKDIR /opt/code-evaluator

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    libc6-dev \
    libstdc++-12-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["fastapi", "run", "app/server.py", "--port", "11451"]
