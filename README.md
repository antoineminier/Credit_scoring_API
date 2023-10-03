##  Project description

This repository contains the backend part files of a machine learning web application which predicts the probabiliy for a client to repay a loan. The application also provides descriptive information to compare the client's data with those of the other loan applicants. The backend part is developped using FastAPI, and the frontend part, using Streamlit.

This project of credit scoring web application is the seventh of the OpenClassrooms Data Scientist course. 

The data used are those of the [Home Credit Default Risk Kaggle's competition](https://www.kaggle.com/c/home-credit-default-risk/data).

See also [the frontend part repository](https://github.com/antoineminier/Credit_scoring_frontend).


## Files description

main.py : the FastAPi code of the backend part of our application

preprocessor.joblib : the pipeline used in main.py to preprocess the client's data

classifier.joblib : the classifier used in main.py to predict if the client would repay the loan he applied for

explainer.joblib : the SHAP explainer to calculate the SHAP values of the different features

descriptions.csv : the descriptions of the different features, to display them in the application

current_applications.csv : the data of the current loan applications' clients

test_backend.py : a part of the tests automatically runned to check if the commits or pull requests can be performed

requirements.txt : the list of packages necessary to run the main.py file

in directory .github, the workflow.yml file define the tests automatically runned to check if the commits or pull requests can be performed ; it uses the test_backend.py file
