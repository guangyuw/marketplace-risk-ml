FROM python:3.12-slim

WORKDIR /app

# Only production dependencies — dev/notebook deps live in requirements-dev.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
# artifacts/ must be present before building; run `python -m src.train` first
COPY artifacts/ artifacts/

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
