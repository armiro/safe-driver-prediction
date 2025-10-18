# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspark


# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "safe_driver_prediction"
VOLUME = "kaggle_competition_data"

# COMMAND ----------

train_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/train.csv"
test_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/test.csv"

train_df = spark.read.csv(path=train_path, header=True, inferSchema=True)
test_df = spark.read.csv(path=test_path, header=True, inferSchema=True)

train_df.limit(10).display()
test_df.limit(10).display()

# COMMAND ----------


train_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.train")
test_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.test")


# COMMAND ----------


