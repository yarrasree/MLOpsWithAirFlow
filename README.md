# MLOpsWithAirFlow
some useful commands for Airflow and MLflow start up in windows
1. initialize the db
open wsl
cd /home/
airflow db init
2.start the airflow web server 
open wsl
cd /home/
airflow webserver
3. start the airflow scheduler
open wsl
cd /home/
airflow scheduler

4. start mlflow server
open wsl
cd /home/
mlflow server --backend-store-uri file:///home/yarra/mlruns/ --no-serve-artifacts

5.start mlflow ui
open wsl
cd /home/yarra/
mlflow ui --host 0.0.0.0 --port 8090

6. Access mlflow ui page
http://localhost:8090/

7. Access the airflow ui page
http://localhost:8080/home

useful function reference:
=========================
https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model
https://pycaret.readthedocs.io/en/stable/api/classification.html#pycaret.classification.plot_model
