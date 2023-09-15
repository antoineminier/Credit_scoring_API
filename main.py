import pandas as pd
import numpy as np
from fastapi import FastAPI
from pathlib import Path
import joblib
import re
import shap
from collections import OrderedDict


df = pd.read_csv('/Users/antoine/Documents/Data_scientist/P7/Data/Prepared_data/test.csv')
descriptions_df = pd.read_csv('/Users/antoine/Documents/Data_scientist/P7/Data/Prepared_data/descriptions.csv')

BASE_DIR = Path(__file__).resolve(strict=True).parent
with open(f"{BASE_DIR}/preprocessor.joblib", "rb") as f:
    preprocessor = joblib.load(f)
with open(f"{BASE_DIR}/classifier.joblib", "rb") as f:
    classifier = joblib.load(f)
with open(f"{BASE_DIR}/explainer.joblib", "rb") as f:
    explainer = joblib.load(f)

id_list = df.loc[:, 'SK_ID_CURR'].values.tolist()

preprocessed_features_names = preprocessor.get_feature_names_out()
preprocessed_features_names = [re.sub('num_preprocessor__|encoder__','',s) for s in preprocessed_features_names]

num_cols = [col for col in df.select_dtypes(include=['int', 'float']).columns if col not in ['SK_ID_CURR']]
cat_cols = [col for col in df.select_dtypes(exclude=['int', 'float']).columns]

app = FastAPI()



@app.get("/get_number_of_features")
def get_number_of_features():
    if 'PREDICTION' in df.columns:
        df.drop(columns='PREDICTION', inplace=True)
    return len(df.loc[:, ~df.columns.isin(['SK_ID_CURR'])].columns)

@app.get("/predict/{id}")
def predict(id): 
    try:
        if int(id) in(id_list):
            sample = df.loc[df['SK_ID_CURR']==int(id), ~df.columns.isin(['SK_ID_CURR'])]
            preprocessed_sample = pd.DataFrame(preprocessor.transform(sample), columns=preprocessed_features_names)
            pred = classifier.predict(preprocessed_sample)[0].astype(float)
            pred_proba = classifier.predict_proba(preprocessed_sample)[0][0]
            results = {'prediction': pred, 'probability': pred_proba}
            return results
        else:
            return("The value entered isn't a valid ID.")
    except:
        return("Please enter digits only.")

@app.get("/descriptions")
def get_descriptions():
    descriptions = {}
    for feature in descriptions_df['Column']:
        descriptions[feature] = descriptions_df.loc[descriptions_df['Column']==feature, 'Description'].values[0]
    return descriptions

@app.get("/explain/{id}")
def explain(id):
    sample = df.loc[df['SK_ID_CURR']==int(id), ~df.columns.isin(['SK_ID_CURR'])]
    preprocessed_sample = pd.DataFrame(preprocessor.transform(sample), columns=preprocessed_features_names)
    shap_values = -explainer.shap_values(preprocessed_sample)[0]

    encoded_features_idx = {}
    for cat_col in cat_cols:
        encoded_features_idx[cat_col]=[]
        for i in range(len(preprocessed_features_names)):
            if preprocessed_features_names[i].startswith(cat_col):
                encoded_features_idx[cat_col].append(i)
    cat_features_impact = {}
    for col in cat_cols:
        cat_features_impact[col]=sum(shap_values[encoded_features_idx[col][0]:encoded_features_idx[col][-1]+1])
    features_impact = cat_features_impact.copy()
    for col in num_cols:
        features_impact[col]=shap_values[preprocessed_features_names.index(col)]
    
    data = sample.copy()
    data[cat_cols] = data[cat_cols].fillna('Missing value')
    data[num_cols] = data[num_cols].fillna('Missing value (replaced by median)')
    data = list(data.loc[:, features_impact.keys()].values[0])

    explanation_dict = {'values': list(features_impact.values()),
                        'expected_value':1-explainer.expected_value,
                        'data':data,
                        'feature_names':list(features_impact.keys())}
    return explanation_dict

@app.get("/compare/{id}")
def compare(id):
    sample = df.loc[df['SK_ID_CURR']==int(id), ~df.columns.isin(['SK_ID_CURR'])]
    preprocessed_sample = pd.DataFrame(preprocessor.transform(sample), columns=preprocessed_features_names)
    shap_values = -explainer.shap_values(preprocessed_sample)[0]
    df_preprocessed = pd.DataFrame(preprocessor.transform(df.drop(columns=['SK_ID_CURR'])), columns=preprocessed_features_names)

    encoded_features_idx = {}
    for cat_col in cat_cols:
        encoded_features_idx[cat_col]=[]
        for i in range(len(preprocessed_features_names)):
            if preprocessed_features_names[i].startswith(cat_col):
                encoded_features_idx[cat_col].append(i)
    cat_features_impact = {}
    for col in cat_cols:
        cat_features_impact[col]=sum(shap_values[encoded_features_idx[col][0]:encoded_features_idx[col][-1]+1])
    features_impact = cat_features_impact.copy()
    for col in num_cols:
        features_impact[col]=shap_values[preprocessed_features_names.index(col)]
    sorted_pairs = sorted(features_impact.items(), key=lambda k: abs(k[1]), reverse=True)
    features_impact = dict(OrderedDict(sorted_pairs))

    features_data = []
    df_copy = df.copy()
    df_copy['PREDICTION'] = classifier.predict(df_preprocessed)
    for feature in list(features_impact.keys()):
        if feature in num_cols:
            df_copy[feature].fillna(value=df_copy[feature].median(), inplace=True)
            barchart_dict = pd.DataFrame([['client', df_copy.loc[df_copy['SK_ID_CURR']==int(id), feature].values[0]], 
                                          ['mean — refused', df_copy.groupby(by='PREDICTION')[feature].mean()[1]],
                                          ['mean — granted', df_copy.groupby(by='PREDICTION')[feature].mean()[0]], 
                                          ['general mean', df_copy[feature].mean()]],
                                          columns=['value displayed', feature]).to_dict('split')
            infos = {'feature': feature, 
                     'feature_impact': features_impact[feature],
                     'barchart_dict': barchart_dict}
            features_data.append(infos)
        elif feature in cat_cols:
            values_for_dict = []
            df_copy[feature].fillna(value='Missing value', inplace=True)
            for cat in df_copy[feature].unique():
                df_cat = df_copy.groupby(by=[feature, 'PREDICTION']).count().reset_index()
                for loan_status, pred in zip(['granted', 'refused'], [0, 1]):
                    if len(df_cat.loc[(df_cat[feature]==cat)&(df_cat['PREDICTION']==pred)])==0:
                        count = 0
                    else:
                        count = df_cat.loc[(df_cat[feature]==cat)&(df_cat['PREDICTION']==pred), 'SK_ID_CURR'].values[0]
                    values_for_dict.append([cat, loan_status, count])
            barchart_dict = pd.DataFrame(values_for_dict, columns=['category', 'loan_status', 'count']).to_dict('split')
            infos = {'feature': feature, 
                     'feature_impact': features_impact[feature],
                     'client_value': df_copy.loc[df_copy['SK_ID_CURR']==int(id), feature].values[0],
                     'barchart_dict': barchart_dict}
            features_data.append(infos)
    return features_data