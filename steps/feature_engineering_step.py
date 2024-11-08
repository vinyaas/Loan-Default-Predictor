import pandas as pd 
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    StandardScaling,
    LabelEncoding
)

from zenml import step
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname=s - %(message)s")

@step
def feature_engineering_step(
    df: pd.DataFrame, log_features: list = None, scaling_features: list = None, encoding_features: list = None
) -> pd.DataFrame:
    """
    Performs feature engineering using FeatureEngineer with multiple strategies.
    """
    # Ensure the DataFrame is valid
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Input must be a pandas DataFrame.')

    df_transformed = df.copy()
    
    if log_features:
        log_transformer = FeatureEngineer(LogTransformation(features=log_features))
        df_transformed = log_transformer.apply_feature_engineering(df_transformed)
        logging.info("Log Transformation applied.")
        logging.info(df_transformed.head())

    if scaling_features:
        standard_scaler = FeatureEngineer(StandardScaling(features=scaling_features))
        df_transformed = standard_scaler.apply_feature_engineering(df_transformed)
        logging.info("Standard Scaling applied.")
        logging.info(df_transformed.head())

    if encoding_features:
        label_encoder = FeatureEngineer(LabelEncoding(features=encoding_features))
        df_transformed = label_encoder.apply_feature_engineering(df_transformed)
        logging.info("Label Encoding applied.")
        logging.info(df_transformed.head())

    return df_transformed
