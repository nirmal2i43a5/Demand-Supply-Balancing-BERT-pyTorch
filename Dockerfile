#Base image 
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app


# Install system dependencies (for ML packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt


COPY app /app/app
COPY static/ /app/static/
COPY models /app/models

EXPOSE 8000

ENV MODEL_PATH=/app/models/bert_ner_baseline_v1

#main.py is the main entry point of the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
