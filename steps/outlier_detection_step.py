import logging 
import pandas as pd

from src.outlier_detection import OutlierDetector , ZScoreOutlierDetection
from zenml import step

@step
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Detects and removes outliers using OutlierDetector."""
    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    if df is None:
        logging.info('Recieved a NoneType DataFrame')
        raise ValueError('Input df must be a non null pandas dataframe')
    
    if not isinstance(df , pd.DataFrame):
        logging.error(f"Expected pandas Dataframe , got {type(df)} instead")
        raise ValueError('input df must be a pandas dataframe')
    
    if column_name not in df.columns:
        logging.error(f'column {column_name} does not exist in dataframe')
        raise ValueError(f"column {column_name} does not exist in dataframe")
        #Ensure only numeric columns are passed
    df_numeric = df.select_dtypes(include=[int , float])
    df_categorical = df.select_dtypes(include='object')
    
    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    outliers = outlier_detector.detect_outliers(df_numeric)
    df_outliers = outlier_detector.handle_outliers(df_numeric, method="remove")
    df_cleaned = pd.concat((df_numeric , df_categorical) , axis = 1)

    return df_cleaned