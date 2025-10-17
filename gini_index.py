# Databricks notebook source
# MAGIC %md
# MAGIC ## Model Evaluation: Normalized Gini Coefficient

# COMMAND ----------

# DBTITLE 1,import libraries
import numpy as np
import pandas as pd
import pyspark

# COMMAND ----------

# DBTITLE 1,calculate gini and normalized gini
# implement gini & normalized gini index logic
# credits: https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703


def calculate_gini_index(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Calculate the gini index for input array of prediction probabilities
    Args:
        actuals (np.ndarray): actual values
        preds (np.ndarray): predicted values
    Return: 
        a float number between 0 (random guess) to 0.5 (perfect score)
    """
    assert len(actuals) == len(preds), "actuals and preds must have the same length"

    all = np.asanyarray(
        np.c_[actuals, preds, np.arange(len(actuals))],
        dtype=np.float16
    )
    all = all[np.lexsort((all[:, 2], -all[:, 1]))]
    actual_losses = all[:, 0].sum()
    if actual_losses == 0:
        raise ValueError("No positive values found in the actual set")
    gini_sum = all[:, 0].cumsum().sum() / actual_losses
    gini_sum -= (len(actuals) + 1) / 2.0
    return float(gini_sum / len(actuals))


def calculate_normalized_gini_index(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Calculate normalized gini index for input array of prediction probabilities
    Args:
        actuals (np.ndarray): actual values
        preds (np.ndarray): predicted values
    Return:
        a float number between 0 (random guess) to 1.0 (perfect score)
    """
    gini_index = calculate_gini_index(actuals, preds)
    self_gini_index = calculate_gini_index(actuals, actuals)
    return gini_index / self_gini_index

# COMMAND ----------

# DBTITLE 1,test
actuals = np.array([1, 1, 0, 0, 0, 0, 0, 1])
preds = np.array([1, 1, 1, 0, 0, 0, 0, 1])

print(calculate_gini_index(
    actuals=actuals,
    preds=preds
))

print(calculate_normalized_gini_index(
    actuals=actuals,
    preds=preds
))

# COMMAND ----------

# DBTITLE 1,calculate normalized gini for model outputs
CATALOG = "workspace"
SCHEMA = "safe_driver_prediction"
TABLE_NAME = "freq_model_results"

MODEL_RESULTS_PATH = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

results_df = spark.read.table(MODEL_RESULTS_PATH)
ground_truth = results_df["actuals"]
model_preds = results_df["preds"]

calculate_normalized_gini_index(
    actuals=ground_truth,
    preds=model_preds
)
