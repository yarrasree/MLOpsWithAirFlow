##############################################################################
# Import necessary modules
# #############################################################################
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import mlflow
import mlflow.sklearn
import sys
from datetime import datetime, timedelta
from Model_experiment_pipeline.constants import *
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import sys, traceback
import pandas as pd
import sqlite3
from pycaret.classification import setup, compare_models,create_model
from pycaret.classification import *
import warnings
warnings.filterwarnings("ignore")

###############################################################################
# Define default arguments and DAG
# ##############################################################################
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,7,30),
    'retries' : 0, 
    'retry_delay' : timedelta(seconds=5)
}

Model_experiment_pipeline = DAG(
                        dag_id ='Model_experiment_pipeline', default_args=default_args,
description = 'DAG to run model experiment', schedule_interval='@hourly')


def read_csv_file():
    dataset = pd.read_csv(DATA_DIRECTORY + "cleaned_data.csv")
    # drop the data column as it is not needed for training
    dataset = dataset.drop(['created_date'], axis=1)
    return df

def build_dbs():

    if os.path.isfile(DB_PATH+DB_FILE_NAME):
        print("DB Already Exsist")
        return "DB Exsists"
    else:
        print("Creating Database")
        """ create a database connection to a SQLite database """
        conn = None
        try:

            conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
            print("New DB Created")
        except Error as e:
            print(e)
            return "Error"
        finally:
            if conn:
                conn.close()
                return "DB Created"
             
def read_dataset_from_sql(table_name, db_filepath):
    conn = sqlite3.connect(db_filepath)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def write_dataset_to_sql(df, table_name, db_filepath):
    conn = sqlite3.connect(db_filepath)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def delete_columns(df, columns):
    df = df.drop(columns=columns)
    return df

def select_columns(dataset, columns):
    dataset_cols = dataset[columns]
    return dataset_cols

def convert_dataframe_to_sql(df, table_name, db_filepath):
    conn = sqlite3.connect(db_filepath)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def preprocess_data():
    table_name = 'cleaned_data'  # Specify the input table name
    db_filepath = DB_PATH+DB_FILE_NAME  # Specify the SQLite database file path
    df = pd.read_csv(DATA_DIRECTORY + "cleaned_data.csv")
    # drop the data column as it is not needed for training
    df = df.drop(['created_date'], axis=1)

    # Write the selected data to new table in the SQLite database
    output_table = 'preprocessed_data_tbl'  # Specify the output table name
    write_dataset_to_sql(df, output_table, db_filepath)
    
    # Pass the output table name and database filepath to the next task using XCom
    return output_table

def setup_pycaret():
    dataset1 = read_dataset_from_sql("preprocessed_data_tbl",DB_PATH+DB_FILE_NAME)
    if dataset1 is None:
        # Handle the case when the dataset is None
        # You can raise an exception, return a default value, or perform any other appropriate action
        raise ValueError("The dataset is None. Please provide a valid dataset.")
    print("dataset is ************************************ ")
    # create a mlflow tracking uri at "http://0.0.0.0:8090"
    # set the tracking uri and experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        exp_lead_scoring = setup(data=dataset1,target='app_complete_flag', remove_multicollinearity=True, multicollinearity_threshold=0.95, categorical_features=['city_tier', 'first_platform_c', 'first_utm_medium_c', 'first_utm_source_c'],fold_shuffle=True,session_id=42,n_jobs=3, use_gpu=False,log_experiment=True,   experiment_name='new_setup_pycaret', log_plots=True,log_data=True, verbose=True,log_profile=False)
        mlflow.sklearn.log_model(exp_lead_scoring, 'exp_pycaret_setup_lead_scoring')
    except:
        print("Except at pycaret_experiment")
    print("Setup function is done") 
    print("compare_models =======>started")
    best_model = compare_models(fold = 5,exclude=['gbc','knn','qda', 'dummy', 'svm', 'ada'])
    mlflow.sklearn.log_model(best_model, 'All_with_best_model')
    print("compare_models =======>End")
    #sys.setrecursionlimit(100000)    
    #models_to_compare = ['lr', 'rf', 'xgboost', 'catboost']
    #model_names = ['lightgbm', 'et', 'dt','rf','lr','ridge','lda','nb']
    #models = [create_model(model_name) for model_name in model_names]
    #best_model = compare_models(*models)
    #lightgbm = create_model('lightgbm',fold = 5)
    #Log best model
    #mlflow.sklearn.log_model(lightgbm, 'lightgbm_createModel')
    #plot_model(lightgbm, plot='feature')
    print("DATA_DIRECTORY path is :",DATA_DIRECTORY)
    feature_importance_plot = plot_model(best_model, plot='feature')
    if feature_importance_plot is not None:
       feature_importance_plot.savefig(DATA_DIRECTORY+'feature_importance_plot.png')
       # Log the feature importance plot to MLflow
       mlflow.log_artifact(DATA_DIRECTORY+'/feature_importance_plot.png', 'feature_importance_plot')
    else:
       print("Error: Feature importance plot is None.")

    print("setup_pycaret function is done")
    # select only important features only    
    dataset_cols = dataset1[SIGNIFICANT_FEATURES]
    try:
        exp_lead_scoring = setup(
            data=dataset_cols,
            target='app_complete_flag',
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,
            categorical_features=['city_tier', 'first_platform_c', 'first_utm_medium_c', 'first_utm_source_c'],
            fold_shuffle=True,
            session_id=42,
            n_jobs=-1,
            use_gpu=False,
            log_experiment=True,
            experiment_name='after_remove_feature_Lead_scoring',
            log_plots=True,
            log_data=True,
            verbose=True,
            log_profile=False
            )
        mlflow.sklearn.log_model(exp_lead_scoring, 'Lead_scoring_dropping_features')
        #lightgbm_fs = create_model('lightgbm')
        #mlflow.sklearn.log_model(lightgbm_fs, 'Removed_feature_lightgbm_createModel')
        lightgbm_fs = compare_models(fold = 10,exclude=['gbc','knn','qda', 'dummy', 'svm', 'ada'])
        mlflow.sklearn.log_model(lightgbm_fs, 'All_with_best_model_dropping_features')
        # Tune the hyper parameters of the lightgbm model using optuna on 10 folds and optimise AUC as that was our system metric, 
        # hence we will optimise AUC
        tuned_lgbm_optuna, tuner_1 = tune_model(
        lightgbm_fs,
        search_library='optuna',
        fold=10,
        optimize='auc',
        choose_better=True,
        return_tuner=True
        )
        with mlflow.start_run():
            mlflow.sklearn.log_model(tuned_lgbm_optuna, 'tuned_lgbm_optuna')
        print(tuned_lgbm_optuna)
    except:
        print("Except at pycaret_experiment")
    
    print("pycaret_experiment function is done")
     

def pycaret_experiment():
    #exp_lead_scoring = kwargs['task_instance'].xcom_pull(task_ids='setup_pycaret')
    dataset = read_dataset_from_sql("preprocessed_data_tbl",DB_PATH+DB_FILE_NAME)

    dataset_cols = dataset[SIGNIFICANT_FEATURES]
#    mlflow.set_tracking_uri(TRACKING_URI)
    #mlflow.set_experiment("pycaret_experiment")
    
    try:
        
        exp_lead_scoring = setup(
            data=dataset,
            target='app_complete_flag',
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,
            categorical_features=['city_tier', 'first_platform_c', 'first_utm_medium_c', 'first_utm_source_c'],
            fold_shuffle=True,
            session_id=42,
            n_jobs=-1,
            use_gpu=False,
            log_experiment=True,
            experiment_name='Yarra_Lead_scoring',
            log_plots=True,
            log_data=True,
            verbose=True,
            log_profile=False
            )
        mlflow.pycaret.log_model(exp_lead_scoring, 'removed_featuress_exp_pycaret_setup_lead_scoring')
    except:
        print("Except at pycaret_experiment")
    print("pycaret_experiment function is done")
    #sys.setrecursionlimit(100000)
    #best_model = compare_models()
    #model_names = ['lightgbm', 'et', 'dt','rf','lr','ridge','lda','nb']
    #models = [create_model(model_name) for model_name in model_names]
    #best_model = compare_models(*models)
    #mlflow.pycaret.log_model(best_model, 'removed_featuress_best_model_comparasion')
    #print("Best Model is :",best_model)
    lightgbm = create_model('lightgbm')
    #Log best model
    mlflow.sklearn.log_model(lightgbm, 'Removed_feature_lightgbm_createModel')
    #plot_model(lightgbm, plot='feature')
    # Log feature importance plot
    #mlflow.sklearn.log_model(lightgbm, plot='feature')

def create_lightGBM_model():
    lightgbm = create_model('lightgbm',fold=5)
    #Log best model
    mlflow.pycaret.log_model(lightgbm, 'lightgbm_createModel')
    #plot_model(lightgbm, plot='feature')
    # Log feature importance plot
    #mlflow.pycaret.plot_model(lightgbm, plot='feature')
    
    
def tuned_lgbm_optuna():
    #output_table = kwargs['task_instance'].xcom_pull(task_ids='preprocess_data')
    # !pip install optuna==2.10.1  # Install Optuna version 2.10.1
    lightgbm_fs = create_model('lightgbm', fold=10)
    tuned_lgbm_optuna, tuner_1 = tune_model(
        lightgbm_fs,
        search_library='optuna',
        fold=10,
        optimize='auc',
        choose_better=True,
        return_tuner=True
    )
    mlflow.sklearn.log_model(lightgbm_fs, 'tuned_lgbm_optuna')
    print(tuned_lgbm_optuna)


preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag = Model_experiment_pipeline
)

setup_pycaret_task = PythonOperator(
    task_id='setup_pycaret',
    python_callable=setup_pycaret,
    provide_context=True,
    dag = Model_experiment_pipeline
)

# Add more tasks

preprocess_task >> setup_pycaret_task
