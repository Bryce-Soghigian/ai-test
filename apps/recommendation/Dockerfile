FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and create necessary directories
COPY src/ src/
COPY models/ models/
RUN mkdir -p logs metrics

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=3002

# Expose port
EXPOSE $PORT

# Copy entrypoint script
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Run the service
ENTRYPOINT ["./docker-entrypoint.sh"] 