FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY config.py .
COPY tensorflow_model.py .
COPY utils.py .
COPY api/ ./api/
COPY models/ ./models/
COPY data/ ./data/
COPY templates/ ./templates/

RUN mkdir -p static logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "app.py"]