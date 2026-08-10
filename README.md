<div align="center">

# 🛒 E-Commerce Conversion Prediction API

**A production-ready MLOps web service predicting real-time user purchase intent.**  
*Built with FastAPI, XGBoost, and modern deployment practices.*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Pipeline-FF9900?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Containerized-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Status-Live%20on%20Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://ecommerce-ml-api-gqub.onrender.com/docs)

</div>

---
> **⚠️ Notice for Evaluators:** This API is hosted on a free cloud instance. If it has been inactive, **it may take approximately 50 seconds to spin up and respond to the first request**. Please be patient on your first call! ⏱️
## 🚀 Live Demo & Documentation
Test the live API interactively through Swagger UI:
> **🔗 [Access the Live API Documentation (/docs)](https://ecommerce-ml-api-gqub.onrender.com/docs#/Predictions/predict_predict_post)**

---

## 📌 Project Overview
In modern e-commerce, accurately predicting whether a browsing session will result in a purchase allows platforms to optimize real-time engagement. This project covers the **complete end-to-end MLOps lifecycle**—from raw data transformation and pipeline training to a secure, documented, and deployed production API.

### ✨ Key Features
* **ML Pipeline Integration**: Leverages a serialized **Scikit-Learn / XGBoost Pipeline** to guarantee consistency between training and inference features.
* **Strict Input Validation**: Uses Pydantic to enforce data contracts (e.g., preventing negative interaction counts via `ge=0` and validating allowed countries).
* **Production Monitoring**: Includes a dedicated `/health` health-check endpoint for container orchestrators.
* **Robust Error Handling**: Structured HTTP exceptions (`400 Bad Request`, `500 Internal Server Error`) to safely handle anomalies.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3.9+
* **API Framework:** FastAPI, Uvicorn
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas, Joblib
* **MLOps & DevOps:** Docker, Git/GitHub, Render

---

## 📁 Project Structure

```text
ecommerce_ml_api/
│
├── main.py                          # FastAPI application endpoints and prediction logic
├── xgboost_ecommerce_pipeline.pkl   # Serialized trained machine learning pipeline
├── requirements.txt                 # Project python dependencies
├── Dockerfile                       # Instructions for building the Docker container image
├── docker-compose.yml               # Multi-container orchestration setup for local deployment
└── README.md                        # Project documentation
