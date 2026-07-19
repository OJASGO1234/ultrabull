# Ultrabull API container. The React frontend is built and served OUTSIDE the
# container (Caddy file_server); this image is the FastAPI backend only.
#
# Build:  docker build -t ultrabull .
# Run:    docker run -d --name ultrabull --restart unless-stopped \
#           -p 127.0.0.1:8000:8000 \
#           --env-file /home/ultrabull/.env -e ULTRABULL_AUTH=1 \
#           -v /home/ultrabull/data_cache:/app/data_cache \
#           ultrabull
# Users:  docker exec -it ultrabull python manage.py adduser <name>
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# numba/vectorbt occasionally need a compiler when a wheel is missing
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
