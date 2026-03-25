FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements_backend.txt

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker api.app:app --bind 0.0.0.0:${PORT} --workers 1"]