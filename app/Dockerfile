FROM python:3.10-slim

WORKDIR /machine-learning-app



COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

COPY app.py app.py

# Run  streamlit with the default attributs
CMD ["streamlit", "run", "app/app.py"]