FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir "fastapi[standard]==0.123.5" psutil==7.2.2

RUN python3 -m nltk.downloader \
    punkt punkt_tab \
    wordnet omw-1.4 \
    stopwords averaged_perceptron_tagger_eng

WORKDIR /app

COPY ./dist/sieval-0.5.0-py3-none-any.whl /tmp/
RUN pip install /tmp/sieval-0.5.0-py3-none-any.whl && rm /tmp/sieval-0.5.0-py3-none-any.whl

COPY submodules /app/submodules
