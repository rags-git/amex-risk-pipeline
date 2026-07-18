from setuptools import setup, find_packages

setup(
    name="amex-credit-risk-engine",
    version="1.0.0",
    description="Enterprise MLOps credit default prediction pipeline with SHAP explainability and FastAPI inference gateway.",
    author="Raghav Bhardwaj",
    packages=find_packages(include=["src", "src.*", "backend", "backend.*"]),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "numpy<2.4",
        "pyarrow>=12.0.0",
        "lightgbm>=3.3.5",
        "scikit-learn>=1.2.0",
        "shap>=0.42.0",
        "mlflow>=2.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "httpx>=0.24.0",
            "flake8>=6.0.0",
        ],
    },
)
