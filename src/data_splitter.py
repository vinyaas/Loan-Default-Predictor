import logging
from abc import ABC , abstractmethod

import pandas as pd 
from sklearn.model_selection import train_test_split

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Data Splitting Strategy
# -----------------------------------------------
# This class defines a common interface for different data splitting strategies.
# Subclasses must implement the split_data method.
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self , df:pd.DataFrame , target_column:str):
        """
        Abstract method to split the data into training and testing sets.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        pass
    
# Concrete Strategy for Simple Train-Test Split
# ---------------------------------------------
# This strategy implements a simple train-test split.
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initializes the SimpleTrainTestSplitStrategy with specific parameters.

        Parameters:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        """
        self.test_size = test_size
        self.random_state = random_state
        
    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Splits the data into training and testing sets using a simple train-test split.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Performing simple train-test split.")
        x = df.drop(columns=[target_column])
        y = df[target_column]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )

        logging.info("Train-test split completed.")
        return x_train, x_test, y_train, y_test
    
    
# Context Class for Data Splitting
# --------------------------------
# This class uses a DataSplittingStrategy to split the data.
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a specific data splitting strategy.

        Parameters:
        strategy (DataSplittingStrategy): The strategy to be used for data splitting.
        """
        self._strategy = strategy

    def set_strategy(self , strategy: DataSplittingStrategy):
        """
        Sets a new strategy for the DataSplitter.

        Parameters:
        strategy (DataSplittingStrategy): The new strategy to be used for data splitting.
        """
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy
        
    def split(self, df: pd.DataFrame, target_column: str):
        """
        Executes the data splitting using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Splitting data using the selected strategy.")
        return self._strategy.split_data(df, target_column)
    
    
if __name__ == "__main__":
    
    file_path =  r'C:\Users\10714194\OneDrive - LTIMindtree\Desktop\py\Machine Learning Tutorial\Loan Defaultor Predictor\extracted_data\feature_engineered.csv'
    df = pd.read_csv(file_path)
    print(df.columns)

    # Initializing data splitter with a specific strategy
    data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    x_train, x_test, y_train, y_test = data_splitter.split(df, target_column='loan_status')
    print('Data split done successfully')
    print(x_train.columns)
    
    logging.info('Saving the train and test datasets to a csv file')
    try:
        x_train.to_csv('C:\\Users\\10714194\\OneDrive - LTIMindtree\\Desktop\\py\\Machine Learning Tutorial\\Loan Defaultor Predictor\\extracted_data\\x_train.csv', index= False)
        x_test.to_csv('C:\\Users\\10714194\\OneDrive - LTIMindtree\\Desktop\\py\\Machine Learning Tutorial\\Loan Defaultor Predictor\\extracted_data\\x_test.csv', index= False)  
        y_train.to_csv('C:\\Users\\10714194\\OneDrive - LTIMindtree\\Desktop\\py\\Machine Learning Tutorial\\Loan Defaultor Predictor\\extracted_data\\y_train.csv', index= False)  
        y_test.to_csv('C:\\Users\\10714194\\OneDrive - LTIMindtree\\Desktop\\py\\Machine Learning Tutorial\\Loan Defaultor Predictor\\extracted_data\\y_test.csv', index= False)  
          
        logging.info(f"Updated dataset saved successfully under extracted_data folder ")    
    except Exception:
        print('Failed to save the updated dataset')
