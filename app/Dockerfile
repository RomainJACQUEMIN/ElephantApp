FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
COPY app/app.py app.py

CMD ["streamlit", "run", "app.py"]
