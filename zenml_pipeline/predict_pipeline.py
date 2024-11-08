import logging
import dill
from zenml import Model, pipeline, step
from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.outlier_detection_step import outlier_detection_step
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info('Pipeline started')

@pipeline(
    model=Model(name='Loan Default Predictor'),
    enable_cache=False
)
def ml_pipeline():
    '''
    Define an end-to-end machine learning pipeline
    '''
    try:
        raw_data = data_ingestion_step(file_path=r'C:\Users\10714194\OneDrive - LTIMindtree\Desktop\py\Machine Learning Tutorial\Loan Defaultor Predictor\data\credit_risk_dataset.zip')
        
        filled_data = handle_missing_values_step(raw_data)
        
        clean_data = outlier_detection_step(filled_data, column_name='person_age')
        
        engineered_data = feature_engineering_step(
            df=clean_data, 
            log_features=['person_income', 'person_age', 'person_emp_length'], 
            scaling_features=['loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'], 
            encoding_features=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        )
        
        x_train, x_test, y_train, y_test = data_splitter_step(engineered_data, target_column='loan_status')
        
        model = model_building_step(x_train, y_train)
        evaluation_metrics, mse = model_evaluator_step(trained_model=model, x_test=x_test, y_test=y_test)
        
        return model

    except Exception as e:
        logging.error(f"Pipeline Execution failed: {e}") 

if __name__ == "__main__":
    run = ml_pipeline()
    logging.info('Pipeline ran successfully')