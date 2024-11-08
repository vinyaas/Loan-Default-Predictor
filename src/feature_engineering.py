import logging
from abc import ABC , abstractmethod
import pickle
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler , LabelEncoder , StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature engineering strategies.
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self , df:pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass
    
# Concrete Strategy for Log Transformation
# ----------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize the distribution.
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self , features):
        """
        Initializes the LogTransformation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f'Applying log transformation to features : {self.features}')
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature]) # log1p handles log(0) by calculating log(1+x)
        logging.info('Log Transformation completed')
        return df_transformed
    
# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()
        
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features : {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info('Standard scaling completed')
        return df_transformed
    
# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)
        
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(
            f'Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}'
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info('Min-Max scaling completed')
        return df_transformed
    
# Concrete Strategy for One-Hot Encoding
# --------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into binary vectors.
class LabelEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features = features
        self.encoder = LabelEncoder()
        
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f'Applying label encoding to features : {self.features}')
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = self.encoder.fit_transform(df[feature])
        logging.info('Label encoding completed')
        return df_transformed
    
# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEngineeringStrategy to apply transformations to a dataset.
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy
        
    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy
        
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)
    
if __name__ == "__main__":
    
    file_path =  r'C:\Users\10714194\OneDrive - LTIMindtree\Desktop\py\Machine Learning Tutorial\Loan Defaultor Predictor\extracted_data\removed_outliers.csv'
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(file_path)

    # Log Transformation
    log_transformer = FeatureEngineer(LogTransformation(features=['person_income', 'person_age' , 'person_emp_length']))
    df_log_transformed = log_transformer.apply_feature_engineering(df)
    print(df_log_transformed)
    with open('log_transformer.pkl', 'wb') as f:
        pickle.dump(log_transformer, f)
    # Standard Scaling
    standard_scaler = FeatureEngineer(StandardScaling(features=['loan_amnt', 'loan_int_rate' , 'loan_percent_income' , 'cb_person_cred_hist_length']))
    df_standard_scaled = standard_scaler.apply_feature_engineering(df_log_transformed)
    print(df_standard_scaled)
    with open('standard_scaler.pkl', 'wb') as f:
        pickle.dump(standard_scaler, f)

    # Label Encoding 
    label_encoder = FeatureEngineer(LabelEncoding(features=['person_home_ownership' , 'loan_intent' , 'loan_grade' , 'cb_person_default_on_file' ]))
    df_label_encoded = label_encoder.apply_feature_engineering(df_standard_scaled)
    print(df_log_transformed)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(df.columns)
    df_final = df_label_encoded
    print(df_final.columns)
    logging.info('Saving the cleaned data as a .csv file')
    try:
        df_final.to_csv('C:\\Users\\10714194\\OneDrive - LTIMindtree\\Desktop\\py\\Machine Learning Tutorial\\Loan Defaultor Predictor\\extracted_data\\feature_engineered.csv', index= False)  
        logging.info(f"Updated dataset saved successfully under extracted_data folder ")    
    except Exception:
        print('Failed to save the updated dataset')
