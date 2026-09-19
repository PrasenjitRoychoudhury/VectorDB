FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

ENV VECTORDB_DATA_DIR=/var/lib/vectordb
VOLUME ["/var/lib/vectordb"]

EXPOSE 8080

CMD ["uvicorn", "vectordb.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
