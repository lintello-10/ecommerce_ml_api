from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Load the XGBoost pipeline model
model = joblib.load("xgboost_ecommerce_pipeline.pkl")

app = FastAPI(title="E-commerce ML API", version="1.0")

# 1. Country mapping dictionary used during training
COUNTRY_MAPPING = {
    "Bahrain": 0,
    "Macao": 1,
    "Malta": 2,
    "Lebanon": 3,
    "Costa Rica": 4,
    "Mongolia": 5,
    "Dominican Republic": 6,
    "Kazakhstan": 7,
    "Georgia": 8,
    "Nigeria": 9
}

# 2. Update input data structure to accept the country name as a string
class EcommerceData(BaseModel):
    count_view_item: int
    count_add_to_cart: int
    count_begin_checkout: int
    device_category: str
    traffic_medium: str
    country: str  # <--- Now accepts the country name directly!

@app.get("/")
def home():
    return {"message": "E-commerce MLOps API is up and running! 🚀"}

@app.post("/predict")
def predict(data: EcommerceData):
    # 3. Convert the country name to its encoded integer value
    # If the country is unknown, default to 0 (or handle the error)
    encoded_country = COUNTRY_MAPPING.get(data.country, 0)
    
    # 4. Prepare the DataFrame with the exact column names expected by the model
    input_data = pd.DataFrame([{
        "count_view_item": data.count_view_item,
        "count_add_to_cart": data.count_add_to_cart,
        "count_begin_checkout": data.count_begin_checkout,
        "device_category": data.device_category,
        "traffic_medium": data.traffic_medium,
        "country_encoded": encoded_country  # <--- Injected here for the model
    }])
    
    # Perform prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    buyer_probability = float(prediction_proba[0][1] * 100)
    is_buyer = int(prediction[0])
    
    return {
        "prediction": is_buyer,
        "buyer_probability_percent": round(buyer_probability, 2),
        "status": "High Conversion" if is_buyer == 1 else "Low Conversion"
    }