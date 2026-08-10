<div align="center">

# 🛒 E-Commerce Conversion Prediction API & MLOps Architecture

**A production-ready MLOps web service predicting real-time user purchase intent, backed by a decoupled Streamlit frontend.**  
*Built with FastAPI, XGBoost, Docker, Streamlit, and modern deployment practices.*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://YOUR_STREAMLIT_APP_URL_HERE)
[![XGBoost](https://img.shields.io/badge/XGBoost-Pipeline-FF9900?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Containerized-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Status-Live%20on%20Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://ecommerce-ml-api-gqub.onrender.com/docs)

</div>

---
> **⚠️ Notice for Evaluators:** The backend API is hosted on a free cloud instance. If it has been inactive, **it may take approximately 50 seconds to spin up and respond to the first request**. Please be patient on your first call! ⏱️

## 🚀 Live Demos & Documentation
* **🔗 [Test the Live API Documentation (Swagger UI)](https://ecommerce-ml-api-gqub.onrender.com/docs#/Predictions/predict_predict_post)**
* **🌐 [Access the Live Streamlit Frontend App](https://ecommercemlapi-nb2638m2scw3u8ztmtts4q.streamlit.app/)**

---

## 📌 Project Overview
In modern e-commerce, accurately predicting whether a browsing session will result in a purchase allows platforms to optimize real-time engagement. This project covers the **complete end-to-end MLOps lifecycle**—from raw data transformation and pipeline training to a secure production API, containerization, and a decoupled user interface.

### 🏛️ Architecture Evolution: V1 vs V2
* **Version 1 (Monolithic Approach):** In the previous iteration, the Streamlit application directly loaded the heavy machine learning model (`.pkl`) into its own local memory to perform predictions. While functional for local testing, it tightly couples the frontend interface with computation logic.
* **Version 2 (Decoupled MLOps Approach - Current):** The architecture has been re-engineered into a professional client-server model. The ML model is wrapped inside a secure **FastAPI** service, containerized via **Docker**, and deployed on **Render**. The new **Streamlit frontend (`app.py`)** acts strictly as a lightweight user interface that communicates with the cloud backend in real-time via HTTP requests (`requests.post`), ensuring scalability and clean separation of concerns.

### ✨ Key Features
* **ML Pipeline Integration**: Leverages a serialized **Scikit-Learn / XGBoost Pipeline** to guarantee consistency between training and inference features.
* **Strict Input Validation**: Uses Pydantic to enforce data contracts (e.g., preventing negative interaction counts via `ge=0` and validating allowed countries).
* **Decoupled Frontend**: Interactive Streamlit dashboard interacting remotely with the cloud API.
* **Production Monitoring**: Includes a dedicated `/health` health-check endpoint for container orchestrators.
* **Robust Error Handling**: Structured HTTP exceptions (`400 Bad Request`, `500 Internal Server Error`) to safely handle anomalies.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3.9+
* **API Framework:** FastAPI, Uvicorn
* **Frontend UI:** Streamlit, Requests
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas, Joblib
* **MLOps & DevOps:** Docker, Git/GitHub, Render

---

## 📁 Project Structure

```text
ecommerce_ml_project/
│
├── api/                             # Backend MLOps service
│   ├── main.py                      # FastAPI application endpoints and prediction logic
│   ├── xgboost_ecommerce_pipeline.pkl # Serialized trained machine learning pipeline
│   ├── requirements.txt             # API python dependencies
│   ├── Dockerfile                   # Instructions for building the Docker container image
│   └── docker-compose.yml           # Multi-container orchestration setup for local deployment
│
├── app.py                           # Version 2 Streamlit web interface (consumes remote API)
├── requirements_app.txt             # Streamlit app dependencies
└── README.md                        # Project documentation