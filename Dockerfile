FROM python:3.12.7-bookworm

ENV PYTHONUNBUFFERED True

WORKDIR /predictor-app
COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app