import streamlit as st
import matplotlib.pyplot as plt
from src.data import load_data, clean_data
from src.model import predict, load_pipe
import pandas as pd


# Set page configuration
st.set_page_config(
    page_title="Data Visualization Tab title",
    )

st.title("Elephant App with streamlite")
st.header("Machine Learning section")
st.write("In this section, we will explore the how to perform a dataviz.")


# Call the model prediction with some parameters
st.subheader("Model prediction")
if st.button("Perform a prediction"):
    prediction_input = pd.DataFrame({ "Id":[1],
                        "MSSubClass":[60],
                        "MSZoning":["RL"],
                        "LotFrontage":[65.0],
                        "LotArea":[8450],
                        "Street":["Pave"],
                        "LotShape":["Reg"],
                        "LandContour":["Lvl"],
                        "Utilities":["AllPub"],
                        "LotConfig":["Inside"],
                        "LandSlope":["Gtl"],
                        "Neighborhood":["CollgCr"],
                        "Condition1":["Norm"],
                        "Condition2":["Norm"],
                        "BldgType":["1Fam"],
                        "HouseStyle":["2Story"],
                        "OverallQual":[7],
                        "OverallCond":[5],
                        "YearBuilt":[2003],
                        "YearRemodAdd":[2003],
                        "RoofStyle":["Gable"],
                        "RoofMatl":["CompShg"],
                        "Exterior1st":["VinylSd"],
                        "Exterior2nd":["VinylSd"],
                        "MasVnrArea":[196.0],
                        "ExterQual":["Gd"],
                        "ExterCond":["TA"],
                        "Foundation":["PConc"],
                        "BsmtQual":["Gd"],
                        "BsmtCond":["TA"],
                        "BsmtExposure":["No"],
                        "BsmtFinType1":["GLQ"],
                        "BsmtFinSF1":[706],
                        "BsmtFinType2":["Unf"],
                        "BsmtFinSF2":[0],
                        "BsmtUnfSF":[150],
                        "TotalBsmtSF":[856],
                        "Heating":["GasA"],
                        "HeatingQC":["Ex"],
                        "CentralAir":["Y"],
                        "Electrical":["SBrkr"],
                        "1stFlrSF":[856],
                        "2ndFlrSF":[854],
                        "LowQualFinSF":[0],
                        "GrLivArea":[1710],
                        "BsmtFullBath":[1],
                        "BsmtHalfBath":[0],
                        "FullBath":[2],
                        "HalfBath":[1],
                        "BedroomAbvGr":[3],
                        "KitchenAbvGr":[1],
                        "KitchenQual":["Gd"],
                        "TotRmsAbvGrd":[8],
                        "Functional":["Typ"],
                        "Fireplaces":[0],
                        "GarageType":["Attchd"],
                        "GarageYrBlt":[2003.0],
                        "GarageFinish":["RFn"],
                        "GarageCars":[2],
                        "GarageArea":[548],
                        "GarageQual":["TA"],
                        "GarageCond":["TA"],
                        "PavedDrive":["Y"],
                        "WoodDeckSF":[0],
                        "OpenPorchSF":[61],
                        "EnclosedPorch":[0],
                        "3SsnPorch":[0],
                        "ScreenPorch":[0],
                        "PoolArea":[0],
                        "MiscVal":[0],
                        "MoSold":[2],
                        "YrSold":[208],
                        "SaleType":["WD"],
                        "SaleCondition":["Normal"],
                        "SalePrice":[208500],
                        #"PoolQC": [], 'Fence': [], 'FireplaceQu': [], 'MasVnrType': [], 'MiscFeature': [], 'Alley': []
                        })
    
    model = load_pipe("model.pkl")
    preproc = load_pipe("preproc.pkl")


    prediction_outpout = predict(pd.DataFrame(preproc.transform(prediction_input), columns=prediction_input.columns), model )
    st.write(prediction_outpout)


# Display a dataframe
st.subheader("Dataframe")
raw_df = st.dataframe(data=load_data())
st.write(raw_df.columns)




# plt.figure(figsize=(12,8))
# plt.scatter(raw_df.columns)
# st.pyplot(plt)
