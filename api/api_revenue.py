from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import pickle
from sklearn.preprocessing import RobustScaler
import uvicorn

# Columns to scale
columns_to_scale = [
       'weight', 'initial_age', 'average_daily_gain', 'feed_quantity_kg',
       'bicarbonate', 'transparency', 'pond_length',
       'morning_temperature', 'size', 'average_body_weight',
       'survival_rate', 'total_harvested', 'area', 'cycle_duration_days',
       'pond_depth', 'total_seed', 'evening_temperature', 'alkalinity',
       'pond_width', 'morning_salinity', 'target_size', 'evening_do',
       'evening_salinity', 'morning_do'
]

def load_scaler(folder_path):
    """
    Load a saved scaler object from a folder.
    """
    file_path = os.path.join(folder_path, 'selling_price_scaler.pkl')
    with open(file_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def transform_data(data, scaler):
    """
    Scale the data using a given scaler.
    """
    scaled_data = scaler.transform(data.loc[:, columns_to_scale])
    data.loc[:, columns_to_scale] = scaled_data
    return data

# API
app = FastAPI() 

# Load the Randomforest model from the file
with open("model/best_model_rf_revenue.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Scaler
scaler = load_scaler('model/')

class api_data(BaseModel):
    weight: float
    initial_age: float
    average_daily_gain: float
    feed_quantity_kg: float
    bicarbonate: float
    transparency: float
    pond_length: float
    morning_temperature: float
    size: float
    average_body_weight: float
    survival_rate: float
    total_harvested: float
    area: float
    cycle_duration_days: int
    pond_depth: float
    total_seed: int
    evening_temperature: float
    alkalinity: float
    pond_width: float
    morning_salinity: float
    target_size: float
    evening_do: float
    evening_salinity: float
    morning_do: float


@app.get("/")
def home():
    return "Hello, FastAPI up!"    

@app.post("/predict/")
def predict(data: api_data):
    try:
        # Convert data api to dataframe
        df = pd.DataFrame(data.dict(), index=[0])
        # Sort Columns
        df = df[sorted(df.columns)]
        # Standard Scaler
        df = transform_data(df, scaler)
        # Make prediction
        prediction = loaded_model.predict(df)
        # Convert prediction to float64
        prediction = float(prediction[0])
        # Return as JSON response
        return {"Revenue": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_revenue:app", host="0.0.0.0", port=8083, reload=True)

"""
example
{
  "weight": 5500.0,
  "initial_age": 24.5,
  "average_daily_gain": 0.92,
  "feed_quantity_kg": 77734.8,
  "bicarbonate": 186.99,
  "transparency": 40.0,
  "pond_length": 290.18,
  "morning_temperature": 271.65,
  "size": 505.0,
  "average_body_weight": 42.9,
  "survival_rate": 49.73,
  "total_harvested": 248000.0,
  "area": 125000.51,
  "cycle_duration_days": 190,
  "pond_depth": 150.35,
  "total_seed": 800136,
  "evening_temperature": 37.98,
  "alkalinity": 194.12,
  "pond_width": 217.26,
  "morning_salinity": 90.53,
  "target_size": 60.0,
  "evening_do": 35.70,
  "evening_salinity": 205.82,
  "morning_do": 55.75
}




"""
