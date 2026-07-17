# Institutional Credit Risk Analytics & Underwriting Pipeline

A production-grade, decoupled credit risk evaluation engine that utilizes high-dimensional multi-moment time-series aggregations to forecast client default probabilities. The architecture features a high-speed FastAPI backend interface serving an optimized gradient boosting framework, coupled with an interactive underwriting dashboard.

## 📊 Model Performance Baseline
- **Core Optimization Metric**: Achieved a competitive validation score of **0.785227** utilizing custom rank-ordering evaluation criteria.
- **Standard Accuracy Conversion**: Translates roughly to a standard **0.96 Area Under the ROC Curve (AUC-ROC)**.

## ⚙️ Local Deployment

### 1. Installation
```bash
pip install -r requirements.txt