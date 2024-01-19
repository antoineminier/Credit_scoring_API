##  Project description

This repository contains the files of a machine learning web application which predicts the probability for a client to repay a loan. The application also explains how much each of the client's feature determined the result of the algorithm and provides descriptive information to compare the client's data with those of the other loan applicants. 

The backend part is developped using FastAPI, and the frontend part, using Streamlit.

Demo of the Streamlit dashboard :

https://github.com/antoineminier/Credit_scoring_frontend/assets/143601336/eafe156d-56d5-448e-915d-0b9929176e44

This project of credit scoring web application is the seventh I worked on in the context of my data scientist training at OpenClassrooms.

The data used are those of the [Home Credit Default Risk Kaggle's competition](https://www.kaggle.com/c/home-credit-default-risk/data).

Address of the backend API on Render : https://credit-scoring-backend.onrender.com/. 
To test for example the prediction function, which return the client’s default probability and the threshold it must not exceed for the loan to be granted, add `predict/` and then a client id (examples of valid client id : 100001, 100005, 100013, 100028, 100038).
Example : https://credit-scoring-backend.onrender.com/predict/100001

Address of the Streamlit dashboard : https://creditscoringapi-dv3azq5m9h995m66syftfg.streamlit.app/. 
Here you just have to enter a valid client id in the provided box to test the app.


## Files description

modeling.ipynb : the jupyter notebook in which the credit scoring algorithm is developed. From this file are exported the .csv and .joblib files used in the backend part of the API, on Render.

data_drift_report.html : the data drift report between the past loan applications' data and the current loan applications' data.

methodology_note.pdf : describes the methodology applied in the modeling process.

presentation.pdf : a slideshow for a 20 minutes presentation of this project.

In the backend folder :
    
    main.py : the FastAPi code of the backend part of our application
    
    preprocessor.joblib : the pipeline used in main.py to preprocess the client's data
    
    classifier.joblib : the classifier used in main.py to predict if the client would repay the loan he applied for
    
    explainer.joblib : the SHAP explainer to calculate the SHAP values of the different features
    
    descriptions.csv : the descriptions of the different features, to display them in the application
    
    current_applications.csv : the data of the current loan applications' clients
    
    test_backend.py : a part of the tests automatically runned to check if the commits or pull requests can be performed
    
    requirements.txt : the list of packages necessary to run the main.py file

in the frontend folder :
    
    dashboard.py : the file containing the frontend code with Streamlit
    
    requirements.txt : the list of packages and their version necessary to run the dashboard.py file

in the .github folder, the workflow.yml file defines the tests automatically runned to check if the commits or pull requests can be performed ; it uses the test_backend.py file in the backend folder
