import catboost
import mlflow
import pandas as pd
import pyspark

import pyspark.sql.functions as F
from sklearn.model_selection import train_test_split


class FrequencyModeling:
    def __init__(
        self, 
        dataset: pyspark.sql.DataFrame, 
        target_col: str,
        test_size: float,
        mlflow_experiment_name: str
        ):
        """
        constructor for the FrequencyModeling class.
        """
        self.spark = pyspark.sql.SparkSession.builder.getOrCreate()
        self.dataset = dataset
        self.target_col = target_col
        self.test_size = test_size
        self.mlflow_experiment_name = mlflow_experiment_name


    def _convert_to_pandas(self) -> pd.DataFrame:
        """
        convert the dataset to pandas for easier manipulation
        """
        return self.dataset.toPandas()


    def split_target_features(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        split the dataset into target and features.
        """
        pd_df = self._convert_to_pandas()
        features = pd_df.drop(columns=[self.target_col])
        target = pd_df[[self.target_col]]
        return (features, target)


    def split_train_test(self):
        """
        split the dataset into train and test sets for model training
        """
        X, y = self.split_target_features()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size, stratify=y, 
            shuffle=True, random_state=40
        )
        return (X_train, X_test, y_train, y_test)


    def create_frequency_model(self, model_name: str, model_params: dict):
        """
        create a frequency model based on the model name and assign the parameters
        """
        if model_name == "catboost":
            model = catboost.CatBoostClassifier(**model_params)
        elif model_name == "xgboost":
            model = None
        else:
            raise ValueError(
                f"model name must be either 'catboost' or 'xgboost', got {model_name}"
            )
        return model




