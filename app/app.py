import streamlit as st
import matplotlib.pyplot as plt
from src.data import load_data, clean_data
from src.model import predict, load_pipe
import pandas as pd
import requests



# Set page configuration
st.set_page_config(
    page_title="Data Visualization Tab title",
    )

st.title("Elephant App with streamlite")
st.header("Machine Learning section")
st.write("In this section, we will explore the how to perform a dataviz.")


# Call the model prediction with some parameters
st.subheader("Model prediction")
user_input = st.text_input("Enter an integer as the LotArea")
if len(user_input) > 0:
    lot_area = user_input
else :
    lot_area = "8450"

if st.button("Perform a pediction"):
    prediction_input = {
            "Id":"1",
            "MSSubClass":"60",
            "MSZoning":"RL",
            "LotFrontage":"65.0",
            "LotArea":lot_area,
            "Street":"Pave",
            "LotShape":"Reg",
            "LandContour":"Lvl",
            "Utilities":"AllPub",
            "LotConfig":"Inside",
            "LandSlope":"Gtl",
            "Neighborhood":"CollgCr",
            "Condition1":"Norm",
            "Condition2":"Norm",
            "BldgType":"1Fam",
            "HouseStyle":"2Story",
            "OverallQual":"7",
            "OverallCond":"5",
            "YearBuilt":"2003",
            "YearRemodAdd":"2003",
            "RoofStyle":"Gable",
            "RoofMatl":"CompShg",
            "Exterior1st":"VinylSd",
            "Exterior2nd":"VinylSd",
            "MasVnrArea":"196.0",
            "ExterQual":"Gd",
            "ExterCond":"TA",
            "Foundation":"PConc",
            "BsmtQual":"Gd",
            "BsmtCond":"TA",
            "BsmtExposure":"No",
            "BsmtFinType1":"GLQ",
            "BsmtFinSF1":"706",
            "BsmtFinType2":"Unf",
            "BsmtFinSF2":"0",
            "BsmtUnfSF":"150",
            "TotalBsmtSF":"856",
            "Heating":"GasA",
            "HeatingQC":"Ex",
            "CentralAir":"Y",
            "Electrical":"SBrkr",
            "1stFlrSF":"856",
            "2ndFlrSF":"854",
            "LowQualFinSF":"0",
            "GrLivArea":"1710",
            "BsmtFullBath":"1",
            "BsmtHalfBath":"0",
            "FullBath":"2",
            "HalfBath":"1",
            "BedroomAbvGr":"3",
            "KitchenAbvGr":"1",
            "KitchenQual":"Gd",
            "TotRmsAbvGrd":"8",
            "Functional":"Typ",
            "Fireplaces":"0",
            "GarageType":"Attchd",
            "GarageYrBlt":"2003.0",
            "GarageFinish":"RFn",
            "GarageCars":"2",
            "GarageArea":"548",
            "GarageQual":"TA",
            "GarageCond":"TA",
            "PavedDrive":"Y",
            "WoodDeckSF":"0",
            "OpenPorchSF":"61",
            "EnclosedPorch":"0",
            "3SsnPorch":"0",
            "ScreenPorch":"0",
            "PoolArea":"0",
            "MiscVal":"0",
            "MoSold":"2",
            "YrSold":"2008",
            "SaleType":"WD",
            "SaleCondition":"Normal"
            }  

    #requests call api 
    API_URL = "http://127.0.0.1:8000/predict_one"
    headers = {    
                "accept": "application/json",
                "Content-Type": "application/json" 
            }

    response = requests.post(API_URL, headers=headers, json=prediction_input)
    price = float(response.content.decode().replace('[', '').replace(']',''))
    
    st.write(f"Predicted price is: USD {price:.2f}.")
