from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    """Verify that the root endpoint responds successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "E-commerce MLOps API" in response.json()["message"]

def test_health_check():
    """Verify that the health check endpoint confirms the model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_endpoint_valid():
    """Verify that a valid prediction request returns a successful response with expected keys."""
    payload = {
        "count_view_item": 5,
        "count_add_to_cart": 2,
        "count_begin_checkout": 1,
        "device_category": "mobile",
        "traffic_medium": "organic",
        "country": "Bahrain"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "buyer_probability_percent" in data
    assert "status" in data

def test_predict_endpoint_invalid_country():
    """Verify that the API properly rejects an unauthorized country with a 400 status code."""
    payload = {
        "count_view_item": 1,
        "count_add_to_cart": 0,
        "count_begin_checkout": 0,
        "device_category": "desktop",
        "traffic_medium": "cpc",
        "country": "Mars"  # Invalid country name
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400