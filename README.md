# 🛒 E-Commerce Conversion Prediction API

An enterprise-grade, containerized Machine Learning API built to predict user purchase conversion probability in real-time. This project implements a full MLOps workflow from exploratory data analysis and model training to deployment readiness using FastAPI and Docker.

---

## 🚀 Project Overview

In the fast-paced e-commerce sector, identifying potential buyers early during a browsing session allows businesses to optimize engagement and boost conversion rates. This project features:
* **Machine Learning Pipeline**: An optimized **XGBoost Classifier** trained on e-commerce user interaction data.
* **RESTful API**: Built with **FastAPI** to deliver ultra-fast, asynchronous predictions with automated interactive documentation (Swagger UI).
* **Containerization**: Packaged into a lightweight **Docker** container ensuring seamless reproducibility across any environment.

---

## 🛠️ Tech Stack

* **Language**: Python 3.9
* **Framework**: FastAPI, Uvicorn
* **Machine Learning**: XGBoost, Scikit-Learn, Pandas, Joblib
* **DevOps & MLOps**: Docker, Git/GitHub

---

## 📁 Project Structure

```text
ecommerce_ml_api/
│
├── main.py                          # FastAPI application endpoints and prediction logic
├── xgboost_ecommerce_pipeline.pkl   # Serialized trained machine learning pipeline
├── requirements.txt                 # Project python dependencies
├── Dockerfile                       # Instructions for building the Docker container image
└── README.md                        # Project documentation