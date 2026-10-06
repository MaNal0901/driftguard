# DriftGuard

![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)
![Data Versioning](https://img.shields.io/badge/DVC-Data%20Versioning-orange)
![Experiment Tracking](https://img.shields.io/badge/MLflow-Tracking-blue)
![Monitoring](https://img.shields.io/badge/Evidently-Drift%20Detection-green)
![Containerization](https://img.shields.io/badge/Docker-Compose-2496ED)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

An MLOps pipeline for energy consumption forecasting, built to demonstrate a full model lifecycle: data versioning, experiment tracking, automated drift detection, and CI/CD-triggered retraining.

---

## Objective

Most portfolio ML projects stop at "trained a model, got a good score." **DriftGuard** is built around a different question: *what happens to a model after it ships?* 

Forecasting models degrade over time as the real world diverges from the data they were trained on. This project implements an automated monitoring loop that catches that degradation, measures drift, decides whether it matters, and triggers retraining without manual intervention.

The forecasting task itself (predicting hourly energy consumption) serves as a practical vehicle for this loop, prioritizing a clear end-to-end MLOps workflow over hyper-parameter optimization.

---

## Key Features

- 🔄 **End-to-End Pipeline**: Pipeline management and data versioning with DVC.
- 📊 **Experiment Tracking**: Full run tracking and metric logging with MLflow.
- 🔍 **Automated Drift Detection**: Feature and target drift detection using Evidently AI.
- 🚨 **Real-Time Alerts**: Orchestrated monitoring and Telegram alerts via n8n workflows.
- 🤖 **Automated Retraining**: Hands-free retraining triggered via GitHub Actions API (`workflow_dispatch`).
- 🚀 **Model Serving**: High-performance REST API built with FastAPI.
- 📈 **Monitoring Dashboard**: Live visual insights with Streamlit.
- 🐳 **Containerized Deployment**: Fully reproducible environments using Docker & Docker Compose.
- 🧪 **Unit Testing**: Suite of automated tests powered by `pytest`.

---

## How the Pipeline Fits Together
