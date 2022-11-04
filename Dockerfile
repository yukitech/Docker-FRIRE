FROM python:3.9.3-slim

RUN mkdir app
WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && \
    apt-get -y install libpq-dev gcc git postgresql-client && \ 
    rm -rf /var/lib/apt/lists/* && \
    pip install -U pip && \
    pip install -r requirements.txt && \
    pip install git+https://github.com/openai/CLIP.git

COPY static ./static/
COPY templates ./templates/
COPY app.py ./
COPY recipeSearch.py ./
COPY encoder.py ./

# flaskアプリケーションの起動
ENV FLASK_APP /app/app.py
CMD flask run -h 0.0.0.0 -p $PORT