from fastapi import FastAPI, HTTPException, status
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# Load the XGBoost pipeline model
model = joblib.load("xgboost_ecommerce_pipeline.pkl")

app = FastAPI(
    title="E-commerce Conversion Prediction API",
    description="Production-ready MLOps API predicting user purchase intent in real-time.",
    version="1.0.0"
)

# Country mapping dictionary used during training
COUNTRY_MAPPING = {
    "Bahrain": 0, "Macao": 1, "Malta": 2, "Lebanon": 3, "Costa Rica": 4,
    "Mongolia": 5, "Dominican Republic": 6, "Kazakhstan": 7, "Georgia": 8, "Nigeria": 9
}

class EcommerceData(BaseModel):
    count_view_item: int = Field(
        ..., 
        ge=0, 
        example=5, 
        description="Number of items viewed by the user (must be >= 0)"
    )
    count_add_to_cart: int = Field(
        ..., 
        ge=0, 
        example=2, 
        description="Number of items added to the cart (must be >= 0)"
    )
    count_begin_checkout: int = Field(
        ..., 
        ge=0, 
        example=1, 
        description="Number of times checkout process was initiated (must be >= 0)"
    )
    device_category: str = Field(
        ..., 
        example="mobile", 
        description="Device category used (e.g., 'mobile', 'desktop', 'tablet')"
    )
    traffic_medium: str = Field(
        ..., 
        example="organic", 
        description="Traffic source medium (e.g., 'organic', 'cpc', 'referral')"
    )
    country: str = Field(
        ..., 
        example="Bahrain", 
        description=f"Country of origin. Allowed values: {list(COUNTRY_MAPPING.keys())}"
    )

@app.get("/", tags=["General"])
def home():
    """Welcome endpoint returning API status."""
    return {"message": "E-commerce MLOps API is up and running! 🚀"}

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint to ensure model readiness for production orchestrators."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not loaded."
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", tags=["Predictions"])
def predict(data: EcommerceData):
    """
    Predict user purchase conversion probability.
    
    Validates input parameters, encodes categorical values, and passes data to the XGBoost pipeline.
    """
    # Strict validation for country input
    if data.country not in COUNTRY_MAPPING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid country '{data.country}'. Allowed values are: {list(COUNTRY_MAPPING.keys())}"
        )
    
    encoded_country = COUNTRY_MAPPING[data.country]
    
    # Prepare DataFrame matching the exact feature order and naming from model training
    input_data = pd.DataFrame([{
        "count_view_item": data.count_view_item,
        "count_add_to_cart": data.count_add_to_cart,
        "count_begin_checkout": data.count_begin_checkout,
        "device_category": data.device_category,
        "traffic_medium": data.traffic_medium,
        "country_encoded": encoded_country
    }])
    
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        buyer_probability = float(prediction_proba[0][1] * 100)
        is_buyer = int(prediction[0])
        
        return {
            "prediction": is_buyer,
            "buyer_probability_percent": round(buyer_probability, 2),
            "status": "High Conversion" if is_buyer == 1 else "Low Conversion"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference error: {str(e)}"
        )
    }