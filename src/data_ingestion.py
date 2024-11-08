import os
import zipfile
import logging
from abc import ABC, abstractmethod
import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define an abstract class for data ingestion
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Abstract method to ingest data from a given file
        """
        pass

# Implement concrete class for ZIP Ingestion
class ZipIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Extracts the zip file and returns the extracted data as a DataFrame
        """
        # Ensure the file is .zip file
        if not file_path.endswith('.zip'):
            raise ValueError('The provided file is not a .zip file')
        
        extract_path = os.path.abspath('../../extracted_data')
        logging.info(f'Extracting files to {extract_path}')

        # Extract the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logging.info('Data extracted successfully')
        
        # Find the extracted csv file (if any)
        extracted_files = os.listdir(extract_path)
        csv_files = [f for f in extracted_files if f.endswith('.csv')]
        
        if len(csv_files) == 0:
            raise FileNotFoundError('No CSV files found')
        if len(csv_files) > 2:
            raise ('Multiple CSV files found. Please specify which one to use')
        
        # Read the csv file to a DataFrame
        csv_file_path = os.path.join(extract_path, csv_files[0])
        df = pd.read_csv(csv_file_path)
        
        return df

# Implement a factory to create data ingestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        "Returns the appropriate DataIngestor based on file extension"
        if file_extension == '.zip':
            return ZipIngestor()
        else:
            raise ValueError(f'No ingestor available for file extension {file_extension}')

if __name__ == '__main__':
    # Specify the path
    file_path = r'C:\Users\10714194\OneDrive - LTIMindtree\Desktop\py\Machine Learning Tutorial\Loan Defaultor Predictor\data\credit_risk_dataset.zip'
    
    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]
    
    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    
    # Ingesting the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)
    print(df.head())
    logging.info('Data Extraction done Successfully')
