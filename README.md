# Amex-Scale Underwriting & Credit Risk Gateway

### Project Overview & Problem Solved
* What it is: An enterprise-grade, end-to-end MLOps pipeline that automates credit default risk evaluation by processing high-dimensional, multi-month customer transaction profiles into real-time risk tiers.
* What it solves: Mitigates financial credit risk by replacing slow, legacy underwriting methods with a low-latency inference gateway that handles malformed client data gracefully via robust Pydantic data firewalls.
* Why it matters: Bridges the gap between raw data and business logic, providing immediate credit limits to low-risk applicants while explaining the exact financial drivers behind high-risk rejections using localized feature attributions.

### Core Technical Skill Matrix
* Machine Learning & Analytics: LightGBM, Stratified K-Fold Cross-Validation, SHAP (Shapley Additive exPlanations), Scikit-Learn, NumPy, Pandas.
* MLOps & Infrastructure: MLflow (Experiment Tracking & Model Registry), Docker (Multi-stage containerization), GitHub Actions (CI/CD automated testing loops).
* Backend Engineering: FastAPI, Uvicorn (Asynchronous ASGI server), Pydantic (Data validation), PyYAML (Decoupled configuration management), Pytest.

### Key Metrics & Engineering Outcomes
* Predictive Performance: Achieved a competitive out-of-fold Gini and normalized top-4% evaluation profile using a custom-engineered evaluation metric explicitly optimized for extreme credit risk variance.
* System Efficiency: Engineered an optimized, chunked data processing pipeline that reduces maximum memory consumption during complex feature aggregations (like payment-to-spend and debt-to-balance velocity ratios).
* Production Reliability: Established 100% automated test coverage across inference entrypoints and built a containerized image that isolates the Python runtime completely, cutting downstream deployment friction.
