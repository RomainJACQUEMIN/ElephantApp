from fastapi import FastAPI
import os
import pandas as pd
from model import load_pipe, predict

app = FastAPI()
preproc = load_pipe("preproc.pkl")
model = load_pipe("model.pkl")

@app.get("/")
def read_root():
    return {"AAAAAAAAAAAH !! L'elephant": f"barit, cours {os.environ.get('NAME')}, {os.environ.get('OTHERNAME')} te poursuit !!! 😱"}

@app.get("/healthcheck")
def is_alive():
    return {"status": "ok"}

@app.post("/predict_one")
def predict_one(data: dict):
    """
    Return the prediction for one house
    """
    return  predict(pd.DataFrame(preproc.transform(data), columns=data.columns), model )

@app.post("/predict_batch")
def predict_batch(data: dict):
    """
    Return a CSV file with the predictions for a batch of houses
    """
    return  predict(pd.DataFrame(preproc.transform(data), columns=data.columns), model )