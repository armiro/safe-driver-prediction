import pyspark
import numpy as np
import pandas as pd
import pyspark.sql.functions as F

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.metrics import pairwise_distances_argmin_min


class DataFactory:
    """
    main data wrangling class containing preprocessing functions
    """
    def __init__(self, df=None):
        """
        initialize the class with constructor
        """
        self.spark = pyspark.sql.SparkSession.builder.getOrCreate()
        self.df = df
    

    def set_df(self, df: pyspark.sql.DataFrame) -> None:
        """
        set the dataframe
        """
        self.df = df
    

    def get_df(self) -> pyspark.sql.DataFrame:
        """
        get the dataframe
        """
        return self.df
    

    def convert_to_pandas(self, df: pyspark.sql.DataFrame) -> pd.DataFrame:
        """
        convert the spark dataframe to a pandas dataframe
        """
        return df.toPandas()
    

    def _extract_feature_columns(self) -> list:
        """
        extract feature columns from the dataframe
        """
        return [col for col in self.df.columns if col not in ["target", "id"]]


    def _split_feature_types(self) -> tuple[list, list, list]:
        """
        extract feature columns and split them by type into 
        categorical, binary and numerical lists
        """
        categorical_cols = []
        binary_cols = []
        numerical_cols = []

        feature_cols = self._extract_feature_columns()
        # we use substring matching since dataset has such standard column names
        for col in feature_cols:
            if col.endswith("_bin"):
                binary_cols.append(col)
            elif col.endswith("_cat"):
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        return (categorical_cols, binary_cols, numerical_cols)


    def standardize_df(self) -> pyspark.sql.DataFrame:
        """
        perform standardization on the dataframe
        """
        new_df = self.df.select(
            [F.col(c).alias(c.replace('ps_', '')) for c in self.df.columns]  # fix column names
        )
        new_df = new_df.select(
            [F.when(F.col(c) == -1, None)  # replace -1 with None for nulls
             .otherwise(F.col(c)).alias(c) for c in new_df.columns]
        )
        self.set_df(new_df)


    def drop_duplicates(self) -> pyspark.sql.DataFrame:
        """
        drop duplicates from the dataframe considering features only
        """
        feature_cols = self._extract_feature_columns()
        new_df = self.df.dropDuplicates(subset=feature_cols)
        self.set_df(new_df)

    
    def _impute_with_prediction(self, df: pyspark.sql.DataFrame, target_col: str) -> pyspark.sql.DataFrame:
        """
        impute missing values in the target column with predictions 
        of a ML model, trained on other features
        """
        null_mask_df = df.filter(F.col(target_col).isNull())  # to be predicted
        not_null_mask_df = df.filter(F.col(target_col).isNotNull())  # used for training
        cat_cols, bin_cols, _ = self._split_feature_types()  # to know target col is cat/bin or not
        null_ids = [row.id for row in null_mask_df.select("id").collect()]  # extract null ids

        input_features = self._extract_feature_columns()
        input_features.remove(target_col)

        X_train = not_null_mask_df.select(input_features).toPandas()
        y_train = not_null_mask_df.select([target_col]).toPandas()
        X_test = null_mask_df.select(input_features).toPandas()

        model_type = "classification" if target_col in [*cat_cols, *bin_cols] else "regression"
        if model_type == "classification":
            model = DecisionTreeClassifier(class_weight="balanced", random_state=40)
        else:
            model = DecisionTreeRegressor(random_state=40)
        
        model.fit(X_train, y_train, sample_weight=None)
        preds = model.predict(X_test)

        spark = pyspark.sql.SparkSession.builder.getOrCreate()
        preds_df = spark.createDataFrame(
            [(idx, float(pred)) for idx, pred in zip(null_ids, preds)], 
            ["id", "predicted_value"]
        )
        joined_df = df.join(preds_df, on="id", how="left")
        imputed_df = joined_df.withColumn(
            target_col,
            F.coalesce(F.col(target_col), F.col("predicted_value"))
        ).drop("predicted_value")

        print(f"imputed high-null column '{target_col}' using {model_type} decision tree model.")
        return imputed_df
        

    def handle_high_null_columns(
        self, 
        threshold: float = 0.5, 
        method: str = "drop", 
        fill_value: int = -1
    ) -> pyspark.sql.DataFrame:
        """
        handle columns with majority of nulls (higher than a threshold)
        """
        if method not in ["drop", "fill", "predict"]:
            raise ValueError(f"method must be 'drop' or 'fill' or 'predict'; got '{method}'")

        total_count = self.df.count()
        feature_cols = self._extract_feature_columns()
        
        high_null_cols = []
        for col in feature_cols:
            null_count = self.df.filter(F.col(col).isNull()).count()
            null_ratio = null_count / total_count
            high_null_cols.append(col) if null_ratio >= threshold else None
        print(f"columns with null ratio higher than {threshold}: {high_null_cols}")

        if method == "drop":
            new_df = self.df.drop(*high_null_cols)
            print(f"dropped {len(high_null_cols)} columns with high null ratio.")
        elif method == "fill":
            new_df = self.df.fillna(fill_value, subset=high_null_cols)
            print(f"filled {len(high_null_cols)} columns with high null ratio, with {fill_value}")
        elif method == "predict":
            new_df = self.df
            for col in high_null_cols:
                new_df = self._impute_with_prediction(df=new_df, target_col=col)
            print(f"imputed {len(high_null_cols)} columns with high null ratio using prediction.")

        self.set_df(new_df)


    def impute_missing_data(self) -> pyspark.sql.DataFrame:
        """
        impute missing data using mean/median/median_log for numerical
        and mode for categorical features
        """
        cat_cols, bin_cols, num_cols = self._split_feature_types()
        for col in self.df.columns:
            if col in [*cat_cols, *bin_cols]:
                mode_val = self.df.groupBy(col).count().orderBy(F.desc("count")).first()
                print(mode_val)
                # TODO: complete the function





