FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN ls -la requirements.txt
RUN cat requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY config.py .
COPY tensorflow_model.py .
COPY utils.py .
COPY api/ ./api/
COPY models/ ./models/
COPY data/ ./data/
COPY templates/ ./templates/
COPY static/ ./static/

RUN mkdir -p static logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "app.py"]
