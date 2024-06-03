from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import pickle
from sklearn.preprocessing import RobustScaler
import uvicorn

# Columns to scale
columns_to_scale = ['target_cultivation_day', 'nitrite', 'evening_temperature', 'evening_salinity', 'alkalinity', 'calcium', 
                    'target_size', 'cycle_duration_days', 'pond_depth', 'area', 'magnesium', 'average_body_weight', 'average_daily_gain', 
                    'pond_width', 'morning_salinity', 'feed_quantity_kg', 'total_seed', 'morning_do', 
                    'bicarbonate', 'morning_temperature', 'pond_length', 'total_plankton_', 'transparency', 'evening_do']

def load_scaler(folder_path):
    """
    Load a saved scaler object from a folder.
    """
    file_path = os.path.join(folder_path, 'survival_rate_scaler.pkl')
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

# Load the XGBoost model from the file
with open("model/best_model_xgb_survival_rate.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Scaler
scaler = load_scaler('model/')

class api_data(BaseModel):
    cycle_duration_days: int
    feed_quantity_kg: float
    alkalinity: float
    average_body_weight: float
    pond_width: float
    area: float
    morning_salinity: float
    total_seed: int
    evening_temperature: float
    morning_temperature: float
    average_daily_gain: float
    transparency: float
    target_size: float
    nitrite: float
    morning_do: float
    pond_depth: float
    target_cultivation_day: float
    pond_length: float
    bicarbonate: float
    evening_do: float
    evening_salinity: float
    magnesium: float
    calcium: float
    total_plankton_: float

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
        return {"Survival_Rate": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_survival_rate:app", host="0.0.0.0", port=8080, reload=True)

"""
example
{
  "cycle_duration_days": 50,
  "feed_quantity_kg": 5000.0,
  "alkalinity": 100.0,
  "average_body_weight": 10.0,
  "pond_width": 50.0,
  "area": 1000.0,
  "morning_salinity": 20.0,
  "total_seed": 50000,
  "evening_temperature": 20.0,
  "morning_temperature": 25.0,
  "average_daily_gain": 1.0,
  "transparency": 30.0,
  "target_size": 50.0,
  "nitrite": 50.0,
  "morning_do": 10.0,
  "pond_depth": 20.0,
  "target_cultivation_day": 60.0,
  "pond_length": 100.0,
  "bicarbonate": 50.0,
  "evening_do": 10.0,
  "evening_salinity": 30.0,
  "magnesium": 100.0,
  "calcium": 50.0,
  "total_plankton_": 500000.0
}


"""
