FROM python:3.10-slim

WORKDIR /api

COPY requirements.txt requirements.txt
COPY src/ src/
COPY setup.py setup.py
COPY models/ models/

RUN pip install -r requirements.txt

COPY api/main.py main.py

# Run the FastAPI app with Uvicorn on $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080","--reload"]
