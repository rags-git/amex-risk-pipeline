# Production-Oriented Credit Risk Underwriting Pipeline

An end-to-end machine learning system for real-time credit risk assessment built with LightGBM, FastAPI, and Docker. The project incorporates automated testing and CI through GitHub Actions while demonstrating scalable data preprocessing, cross-validated model training, and production-oriented deployment practices inspired by enterprise underwriting workflows. The project emphasizes reproducibility, modular software design, and scalable preprocessing for large tabular financial datasets.

### Project Overview & Problem Solved
* What it is: A modular machine learning pipeline built to process high-dimensional credit data and expose a REST API for real-time credit risk inference.
* What it solves: Demonstrates how traditional batch underwriting workflows can be modernized using a real-time ML inference service with robust request validation using Pydantic.
* Why it matters: Bridges the gap between raw data and business logic, providing structured risk scores and automated risk classifications.

### Technical Stack & Architecture
* Core Machine Learning: LightGBM, Stratified K-Fold Cross-Validation, Scikit-Learn, NumPy, Pandas.
* Interface & Service Layer: FastAPI, Uvicorn, Pydantic (data contract validation), PyYAML (decoupled configuration management).
* Containerization & Testing: Multi-stage Docker optimization, Pytest unit-testing framework.

### System Architecture Flow

    CSV Dataset
          │
          ▼
    Chunk-Based Preprocessing
          │
          ▼
    Feature Aggregation
          │
          ▼
    LightGBM Training (5-Fold CV)
          │
          ▼
    Serialized Model
          │
          ▼
    FastAPI Inference Service
          │
          ▼
    Prediction Endpoint
          │
          ▼
    Risk Score & Classification

### Repository Architecture

    .
    ├── .github/workflows/
    │   └── ci.yml               # Automated GitHub Actions test pipeline
    ├── backend/
    │   ├── app.py               # FastAPI inference service
    │   └── schemas.py           # Pydantic request/response models
    ├── config.yaml              # Hyperparameters and pipeline paths configuration
    ├── Dockerfile               # High-performance multi-stage deployment build
    ├── requirements.txt         # Pinned python dependency matrix
    ├── setup.py                 # Structural editable module configuration wrapper
    ├── src/
    │   ├── data_pipeline.py     # Chunk-based preprocessing & feature aggregation
    │   └── ml_engine.py         # LightGBM training & Stratified K-Fold validation
    └── tests/
        └── test_api.py          # Automated endpoint validation and error tests

### Design Decisions & Why LightGBM?
* Tabular Optimization: LightGBM was selected for its exceptional performance on large, high-dimensional tabular financial datasets.
* Structural Features: Efficient histogram-based training algorithms drastically optimize memory overhead, and native handling of missing values natively tracks variable reporting profiles without information leakage.
* Robust Validation: Stratified K-Fold Cross-Validation produces stable, un-skewed out-of-fold validation estimates under extreme class imbalance conditions.
* Footprint Scalability: Chunk-based preprocessing bounds peak active memory consumption, allowing large-scale time-series execution bounds on localized host units.

### Results
| Property | Baseline Specification Metric |
| :--- | ---: |
| Model | LightGBM Booster Engine |
| Validation Strategy | 5-Fold Stratified Cross-Validation |
| Evaluation Metric | Official AMEX Corporate Metric |
| Validation Score | 0.796 |
| ROC-AUC Benchmark | 0.958 |

### Pipeline Engineering & Evaluation
* Operational Metric: Implements the official Amex evaluation metric M = 0.5 * (G + D), which explicitly averages the Normalized Gini Coefficient (G) with the default rate performance at the top 4% sorting threshold (D).
* Data Optimization: Features a chunk-based pre-processing engine designed to stream major transaction files by down-casting numeric memory allocations to reduce operational RAM overhead.
* Resilient Deployment: Containerized via a multi-stage Docker blueprint that separates compilation layers from the active execution image, cutting image weight and enforcing isolated dependency parameters.

### API Endpoint Interface Blueprint
#### Request Definition
`POST /api/v1/predict-risk`

    {
      "customer_id": "0000099db563072c3d5e7",
      "features": {
        "P_2_mean_0": 0.62,
        "D_39_max_0": 0.02,
        "B_1_last_0": 0.01
      }
    }

#### Response Definition

    {
      "customer_id": "0000099db563072c3d5e7",
      "probability_of_default": 0.0642,
      "risk_classification": "Low Risk Profile",
      "recommended_credit_limit": 45000.0,
      "underwriting_action": "AUTO_APPROVE"
    }

### Quick Start

    # Clone the architecture workspace repository
    git clone https://github.com/rags-git/amex-risk-pipeline.git
    cd amex-risk-pipeline

    # Configure and resolve pinned environment dependencies
    pip install -r requirements.txt
    pip install -e .

    # Process raw transactional ingestion loops
    python src/data_pipeline.py

    # Execute the ensembled training validations matrix
    python src/ml_engine.py

    # Launch the live asynchronous local gateway server
    uvicorn backend.app:app --reload