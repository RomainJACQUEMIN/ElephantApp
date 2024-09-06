from fastapi import FastAPI
import os
import pandas as pd
from model import load_pipe, predict

app = FastAPI()
preproc = load_pipe("preprocessor.pkl")
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
    # Cast string values respect schema when possible
    data_formated = {}
    for key,value in data.items():
        try : 
            data_formated[key]= float(value)
        except: 
            data_formated[key]=  value

    data_formated= pd.DataFrame([data_formated])
    result =  list(predict(data_formated, preproc, model))
    return result

@app.post("/predict_batch")
def predict_batch(data: dict):
    """
    Return a CSV file with the predictions for a batch of houses
    """
    return  predict(pd.DataFrame(preproc.transform(data), columns=data.columns), model )