from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import pickle
from sklearn.preprocessing import RobustScaler
import uvicorn

# Columns to scale
columns_to_scale = [
       'total_seed', 'feed_quantity_kg', 'bicarbonate', 'alkalinity',
       'cycle_duration_days', 'average_body_weight', 'evening_do',
       'magnesium', 'pond_length', 'area', 'morning_salinity',
       'morning_temperature', 'average_daily_gain', 'pond_width',
       'target_size', 'evening_salinity', 'evening_temperature',
       'transparency', 'pond_depth', 'nitrite', 'morning_do', 'calcium',
       'ammonia', 'carbonate'
]

def load_scaler(folder_path):
    """
    Load a saved scaler object from a folder.
    """
    file_path = os.path.join(folder_path, 'biomass_scaler.pkl')
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

# Load the Gradientboost model from the file
with open("model/best_model_xgb_biomass.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Scaler
scaler = load_scaler('model/')

class api_data(BaseModel):
    total_seed: int
    feed_quantity_kg: float
    bicarbonate: float
    alkalinity: float
    cycle_duration_days: int
    average_body_weight: float
    evening_do: float
    magnesium: float
    pond_length: float
    area: float
    morning_salinity: float
    morning_temperature: float
    average_daily_gain: float
    pond_width: float
    target_size: float
    evening_salinity: float
    evening_temperature: float
    transparency: float
    pond_depth: float
    nitrite: float
    morning_do: float
    calcium: float
    ammonia: float
    carbonate: float

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
        return {"Biomass": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_biomass:app", host="0.0.0.0", port=8082, reload=True)

"""
example
{
  "total_seed": 750000,
  "feed_quantity_kg": 150000.0,
  "bicarbonate": 200.0,
  "alkalinity": 150.0,
  "cycle_duration_days": 200,
  "average_body_weight": 50.0,
  "evening_do": 30.0,
  "magnesium": 2000.0,
  "pond_length": 400.0,
  "area": 10000.0,
  "morning_salinity": 100.0,
  "morning_temperature": 300.0,
  "average_daily_gain": 1.0,
  "pond_width": 200.0,
  "target_size": 60.0,
  "evening_salinity": 300.0,
  "evening_temperature": 50.0,
  "transparency": 40.0,
  "pond_depth": 150.0,
  "nitrite": 250.0,
  "morning_do": 20.0,
  "calcium": 1000.0,
  "ammonia": 10.0,
  "carbonate": 50.0
}



"""
