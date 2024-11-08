from typing import Tuple

import pandas as pd 
from src.data_splitter import DataSplitter , SimpleTrainTestSplitStrategy
from zenml import step

@step
def data_splitter_step(
    df: pd.DataFrame , target_column : str)-> Tuple[pd.DataFrame , pd.DataFrame , pd.Series , pd.Series]:
    '''
    Splits the data into training and testing sets using datasplitter and a chosen strategy 
    '''
    splitter = DataSplitter(strategy= SimpleTrainTestSplitStrategy())
    x_train , x_test , y_train , y_test = splitter.split(df , target_column)
    return x_train , x_test , y_train , y_test