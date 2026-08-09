from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# Load the XGBoost pipeline model
model = joblib.load("xgboost_ecommerce_pipeline.pkl")

app = FastAPI(title="E-commerce ML API", version="1.0")

COUNTRY_MAPPING = {
    "Bahrain": 0, "Macao": 1, "Malta": 2, "Lebanon": 3, "Costa Rica": 4,
    "Mongolia": 5, "Dominican Republic": 6, "Kazakhstan": 7, "Georgia": 8, "Nigeria": 9
}

class EcommerceData(BaseModel):
    count_view_item: int = Field(..., example=5, description="Number of items viewed by the user")
    count_add_to_cart: int = Field(..., example=2, description="Number of items added to the cart")
    count_begin_checkout: int = Field(..., example=1, description="Number of times checkout process was initiated")
    device_category: str = Field(..., example="mobile", description="Device type: 'mobile', 'desktop', or 'tablet'")
    traffic_medium: str = Field(..., example="organic", description="Traffic source medium: 'organic', 'cpc', 'referral', etc.")
    country: str = Field(..., example="Bahrain", description=f"Country of origin. Accepted values: {list(COUNTRY_MAPPING.keys())}")

@app.get("/")
def home():
    return {"message": "E-commerce MLOps API is up and running! 🚀"}

@app.post("/predict")
def predict(data: EcommerceData):
    # Convert the country name to its encoded integer value
    encoded_country = COUNTRY_MAPPING.get(data.country, 0)
    
    # Prepare the DataFrame
    input_data = pd.DataFrame([{
        "count_view_item": data.count_view_item,
        "count_add_to_cart": data.count_add_to_cart,
        "count_begin_checkout": data.count_begin_checkout,
        "device_category": data.device_category,
        "traffic_medium": data.traffic_medium,
        "country_encoded": encoded_country 
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