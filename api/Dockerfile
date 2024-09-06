FROM python:3.10-slim

WORKDIR /elephant-app

COPY requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt
RUN pip install -e .


COPY main.py main.py

# Run the FastAPI app with Uvicorn on $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080","--reload"]
