import logging 
from abc import ABC , abstractmethod
from joblib import dump
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self , x_train : pd.DataFrame , y_train :pd.Series) -> ClassifierMixin:
        """ 
        Abstract method to build and train a model.
        Parameters: X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target. 
        
        Returns: ClassifierMixin: A trained scikit-learn model instance. 
        """ 
        pass
    
# Concrete Strategy for Random Forest Classification using scikit-learn
class RandomForestStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """ 
        Abstract method to build and train a model. 
        Parameters: X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target. 
        
        Returns: ClassifierMixin: A trained scikit-learn model instance. 
        """ 
        pass
    
# Concrete Strategy for Random Forest Classification using scikit-learn 
class RandomForestStrategy(ModelBuildingStrategy): 
    def build_and_train_model(self, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline: 
        """ 
        Builds and trains a Random Forest classification model using scikit-learn. 
        Parameters: X_train (pd.DataFrame): The training data features. 
        y_train (pd.Series): The training data labels/target. 
        
        Returns: Pipeline: A scikit-learn pipeline with a trained Random Forest model. 
        """
        #Ensure the inputs are of correct type 
        if not isinstance(x_train ,pd.DataFrame ):
            raise TypeError("x_train must be a pandas DataFrame")
        if not isinstance(y_train , pd.Series):
            raise TypeError('y_train must be a pandas series')
        
        logging.info("Initializing Random Forest model with scaling.")
        
        #create a pipeline with random forest model 
        pipeline = Pipeline(
            [
                ('model' , RandomForestClassifier(random_state=42))
            ]
        )
        logging.info('Training Random forest model')
        pipeline.fit(x_train , y_train)
        logging.info("Model training completed")
        return pipeline
    
# Context Class for Model Building 
class ModelBuilder: 
    def __init__(self, strategy: ModelBuildingStrategy):
        """ 
        Initializes the ModelBuilder with a specific model building strategy.
        Parameters: 
        strategy (ModelBuildingStrategy): The strategy to be used for model building. 
        """ 
        self._strategy = strategy 
        
    def set_strategy(self, strategy: ModelBuildingStrategy): 
        """ 
        Sets a new strategy for the ModelBuilder. 
        Parameters: 
        strategy (ModelBuildingStrategy): The new strategy to be used for model building. 
        """ 
        logging.info("Switching model building strategy.") 
        self._strategy = strategy 
        
    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin: 
        """ 
        Executes the model building and training using the current strategy. 
        Parameters: 
        X_train (pd.DataFrame): The training data features. 
        y_train (pd.Series): The training data labels/target. 
        
        Returns: ClassifierMixin: A trained scikit-learn model instance. 
        """ 
        logging.info("Building and training the model using the selected strategy.") 
        return self._strategy.build_and_train_model(X_train, y_train)
    
if __name__ == "__main__": 

    x_train =  pd.read_csv(r'C:\Users\10714194\OneDrive - LTIMindtree\Desktop\py\Machine Learning Tutorial\Loan Defaultor Predictor\extracted_data\x_train.csv')
    y_train =  pd.read_csv(r'C:\Users\10714194\OneDrive - LTIMindtree\Desktop\py\Machine Learning Tutorial\Loan Defaultor Predictor\extracted_data\y_train.csv').squeeze()
    
    model_builder = ModelBuilder(RandomForestStrategy()) 
    model = model_builder.build_model(x_train, y_train) 
    logging.info("Model trained successfully!")
    logging.info('Saving the model  as a joblib file')
    try: 
        with open('model.pkl', 'wb') as file: 
            pickle.dump(model, file) 
            logging.info("Model saved successfully!") 
    except Exception: 
        logging.error(f"Failed to save the model")