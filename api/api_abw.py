from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import pickle
from sklearn.preprocessing import RobustScaler
import uvicorn

# Columns to scale
columns_to_scale = [
      'cycle_duration_days', 'feed_quantity_kg', 'total_seed',
       'initial_age', 'evening_salinity', 'evening_temperature',
       'pond_width', 'morning_temperature', 'pond_length', 'area',
       'evening_do', 'pond_depth', 'morning_do', 'morning_salinity',
       'transparency', 'target_cultivation_day', 'alkalinity',
       'target_size', 'bicarbonate', 'ammonia', 'nitrite', 'carbonate',
       'hardness', 'nitrate'
]

def load_scaler(folder_path):
    """
    Load a saved scaler object from a folder.
    """
    file_path = os.path.join(folder_path, 'abw_scaler.pkl')
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
with open("model/best_model_gb_abw.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Scaler
scaler = load_scaler('model/')

class api_data(BaseModel):
    cycle_duration_days: int
    feed_quantity_kg: float
    total_seed: int
    initial_age: float
    evening_salinity: float
    evening_temperature: float
    pond_width: float
    morning_temperature: float
    pond_length: float
    area: float
    evening_do: float
    pond_depth: float
    morning_do: float
    morning_salinity: float
    transparency: float
    target_cultivation_day: float
    alkalinity: float
    target_size: float
    bicarbonate: float
    ammonia: float
    nitrite: float
    carbonate: float
    hardness: float
    nitrate: float

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
        return {"Average_Body_Weight": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_abw:app", host="0.0.0.0", port=8081, reload=True)

"""
example
{
  "cycle_duration_days": 183,
  "feed_quantity_kg": 109770.25,
  "total_seed": 900636,
  "initial_age": 34.5,
  "evening_salinity": 205.82361746987955,
  "evening_temperature": 37.9840625,
  "pond_width": 44.0,
  "morning_temperature": 271.6526981132075,
  "pond_length": 289.75,
  "area": 20000.51,
  "evening_do": 35.701785,
  "pond_depth": 150.35,
  "morning_do": 302.9346936827957,
  "morning_salinity": 90.5296603773585,
  "transparency": 40.0,
  "target_cultivation_day": 183.5,
  "alkalinity": 194.1166037735849,
  "target_size": 70.0,
  "bicarbonate": 186.99056603773585,
  "ammonia": 11.557865168539324,
  "nitrite": 263.9078791208791,
  "carbonate": 63.75,
  "hardness": 11174.924731182795,
  "nitrate": 14.790697674418604
}



"""
