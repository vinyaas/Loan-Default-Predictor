import logging 
from abc import ABC , abstractmethod

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Outlier Detection Strategy
class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to detect outliers in the given DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features for outlier detection.

        Returns:
        pd.DataFrame: A boolean dataframe indicating where outliers are located.
        """
        pass

# Concrete Strategy for Z-Score Based Outlier Detection
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self , threshold = 3):
        self.threshold = threshold
        
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info('Detecting outliers using zscore method .')
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores>self.threshold
        logging.info(f"Outliers detected with z-score threshold : {self.threshold} ")
        return outliers
    
# Concrete Strategy for IQR Based Outlier Detection
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info('Detecting outliers using IQR method')
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info('Outliers detected using IQR method')
        return outliers 
    
# Context class for outlier detection and handling 
class OutlierDetector:
    def __init__(self , strategy : OutlierDetectionStrategy):
        self._strategy = strategy
        
    def set_strategy(self , strategy : OutlierDetectionStrategy):
        logging.info('Switching outlier detection strategy')
        self._strategy = strategy
        
    def detect_outliers(self , df:pd.DataFrame) -> pd.DataFrame:
        logging.info('Executing outlier detection strategy .')
        return self._strategy.detect_outliers(df)
    
    def handle_outliers(self , df:pd.DataFrame , method = 'remove' , **kwrags) -> pd.DataFrame:
        outliers = self.detect_outliers(df)
        if method == 'remove':
            logging.info('Removing outliers from the dataset')
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == 'cap':
            logging.info('Capping outliers in the dataset')
            df_cleaned = df.clip(lower=df.quantile(0.01) , upper= df.quantile(0.99) , axis = 1)
        else:
            logging.warning(f"Unknown method {method} . No outlier handling performed")
            return df
        
        logging.info('Outlier handling completed')
        return df_cleaned
    
    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")
        
if __name__ == "__main__":
    
    file_path =  r'C:\Users\10714194\OneDrive - LTIMindtree\Desktop\py\Machine Learning Tutorial\Loan Defaultor Predictor\extracted_data\fixed_missing_values.csv'
    df = pd.read_csv(file_path)
    df_numeric = df.select_dtypes(include=[np.number]).dropna()

    # # Initialize the OutlierDetector with the Z-Score based Outlier Detection Strategy
    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))

    # # Detect and handle outliers
    outliers = outlier_detector.detect_outliers(df_numeric)
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")
    
    # # Combining the numerical and categorical columns and realligning the indexes to a final dataset 
    df_categorical = df.select_dtypes(include='object')
    df_categorical = df_categorical.loc[df_cleaned.index]
    final_cleaned_df = pd.concat((df_cleaned , df_categorical) , axis=1)
    logging.info('Saving the cleaned data as a .csv file')
    try:
        final_cleaned_df.to_csv('C:\\Users\\10714194\\OneDrive - LTIMindtree\\Desktop\\py\\Machine Learning Tutorial\\Loan Defaultor Predictor\\extracted_data\\removed_outliers.csv', index= False)  
        logging.info(f"Updated dataset saved successfully under extracted_data folder ")    
    except Exception:
        print('Failed to save the updated dataset')      