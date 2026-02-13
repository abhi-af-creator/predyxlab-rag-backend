FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install CPU-only torch explicitly
RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Then install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker app.main:app \
--bind 0.0.0.0:${PORT:-8000} \
--workers 2 \
--timeout 300 \
--keep-alive 120"]
