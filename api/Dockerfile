FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements_backend.txt

CMD ["gunicorn", "-k", 
"uvicorn.workers.UvicornWorker", 
"api.app:app", "--bind",
 "0.0.0.0:10000"]