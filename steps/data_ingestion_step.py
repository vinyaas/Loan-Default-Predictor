import pandas as pd 
from src.data_ingestion import DataIngestorFactory
from zenml import step

@step
def data_ingestion_step(file_path : str) -> pd.DataFrame:
    '''
    Ingest data from a ZIP file using DataIngestor
    '''
    # Determine the file extension 
    file_extension = '.zip'
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = data_ingestor.ingest(file_path)
    return df