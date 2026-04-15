from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

app = FastAPI(title="Adult Income Prediction API")

model = load_model("best_pipeline")


class PredictionInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    data = pd.DataFrame([{
        "age": input_data.age,
        "workclass": input_data.workclass,
        "fnlwgt": input_data.fnlwgt,
        "education": input_data.education,
        "education-num": input_data.education_num,
        "marital-status": input_data.marital_status,
        "occupation": input_data.occupation,
        "relationship": input_data.relationship,
        "race": input_data.race,
        "sex": input_data.sex,
        "capital-gain": input_data.capital_gain,
        "capital-loss": input_data.capital_loss,
        "hours-per-week": input_data.hours_per_week,
        "native-country": input_data.native_country
    }])

    prediction = predict_model(model, data=data)

    return {"prediction": str(prediction.iloc[0]["prediction_label"])}