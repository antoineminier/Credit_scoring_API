from pathlib import Path
import joblib
import pandas as pd
import re
import numpy as np



class TestBackEnd:

    BASE_DIR = Path(__file__).resolve(strict=True).parent
    with open(f"{BASE_DIR}/preprocessor.joblib", "rb") as f:
        preprocessor = joblib.load(f)
    with open(f"{BASE_DIR}/classifier.joblib", "rb") as f:
        classifier = joblib.load(f)
    with open(f"{BASE_DIR}/explainer.joblib", "rb") as f:
        explainer = joblib.load(f)

    df = pd.read_csv('current_applications.csv')
    descriptions_df = pd.read_csv('descriptions.csv')

    preprocessed_features_names = preprocessor.get_feature_names_out()
    preprocessed_features_names = [re.sub('num_preprocessor__|encoder__','',s) for s in preprocessed_features_names]

    """id 100038 has a NaN in columns OCCUPATION_TYPE and EXT_SOURCE_3 (espectively categorical and numerical) > using this id enables testing of NaN handling"""
    id = '100038'
    sample = df.loc[df['SK_ID_CURR']==int(id), ~df.columns.isin(['SK_ID_CURR'])]
    preprocessed_sample = pd.DataFrame(preprocessor.transform(sample), columns=preprocessed_features_names)

    def test_data_files(self):
        """Check if datasets are empty"""
        assert((len(self.df)>0) & (len(self.descriptions_df)>0))

    def test_preprocessor(self):
        """Check if there are still missing values in the sample after preprocessing"""
        assert(self.preprocessed_sample.isnull().values.any()==False)

    def test_classifier(self):
        pred = self.classifier.predict(self.preprocessed_sample)[0].astype(float)
        pred_proba = self.classifier.predict_proba(self.preprocessed_sample)[0][0]
        assert((pred in [0, 1]) & (pred_proba>=0) & (pred_proba<=1))

    def test_explainer(self):
        shap_values = self.explainer.shap_values(self.preprocessed_sample)[0]
        expected_value = self.explainer.expected_value
        assert((all((value>-1) & (value<1) for value in shap_values))
               & ((expected_value >=0) & (expected_value <=1)))
