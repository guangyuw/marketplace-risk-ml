# Databricks Integration Guide

This project trains locally with MLflow. To run on Databricks (Community Edition or a trial workspace), follow the steps below.

---

## 1. Upload project

1. Clone or zip `marketplace-risk-ml/`
2. In Databricks: **Workspace → Import** the `src/` folder
3. Or attach this repo via Git integration (Repos)

---

## 2. Notebook training

Create a notebook `train_marketplace_risk` with:

```python
# Databricks notebook source
%pip install xgboost mlflow matplotlib
dbutils.library.restartPython()

import sys
sys.path.insert(0, "/Workspace/Repos/<you>/marketplace-risk-ml")

from src.train import train_pipeline

result = train_pipeline(register_model=False)
display(result)
```

MLflow runs appear under **Experiments** in the Databricks UI (managed MLflow).

---

## 3. Register model

After a successful training run:

```python
import mlflow

run_id = result["run_id"]
mv = mlflow.register_model(
    model_uri=f"runs:/{run_id}/xgb_model",
    name="marketplace_risk_scorer"
)
print(f"Version: {mv.version}")
```

Navigate to **Models → marketplace_risk_scorer** in the Databricks UI to see the registered version.

---

## 4. Model Serving endpoint

1. In the Databricks UI: **Serving → Create serving endpoint**
2. Select model `marketplace_risk_scorer`, choose a version
3. Select CPU / small instance type → Create

The endpoint provides a REST API equivalent to the local FastAPI `/predict` endpoint, with Databricks managing autoscaling and availability.

---

## 5. Load from Snowflake (optional)

```python
# After running sql/feature_extraction_snowflake.sql in Snowflake
df = (spark.read
      .format("snowflake")
      .option("sfUrl", "<account>.snowflakecomputing.com")
      .option("sfDatabase", "MARKETPLACE")
      .option("sfSchema", "ANALYTICS")
      .option("sfWarehouse", "COMPUTE_WH")
      .option("dbtable", "FEATURES_TRAINING")
      .load()
      .toPandas())
```

Replace `generate_synthetic_transactions()` in `train.py` with `df` for real data.

---

## 6. Delta Lake

Historical features and labels stored in Delta tables support ACID updates and time-travel — useful when retraining after policy changes or retroactive label corrections.

---

## Resources

- [Databricks Community Edition](https://community.cloud.databricks.com/)
- [Snowflake Free Trial](https://signup.snowflake.com/)
- [MLflow Model Registry docs](https://mlflow.org/docs/latest/model-registry.html)
