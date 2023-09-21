from pathlib import Path
import joblib
import pandas as pd
import re



class TestBackEnd:

    BASE_DIR = Path(__file__).resolve(strict=True).parent
    with open(f"{BASE_DIR}/preprocessor.joblib", "rb") as f:
        preprocessor = joblib.load(f)
    with open(f"{BASE_DIR}/classifier.joblib", "rb") as f:
        classifier = joblib.load(f)
    with open(f"{BASE_DIR}/explainer.joblib", "rb") as f:
        explainer = joblib.load(f)

    df = pd.read_csv('test.csv')

    id_list = df.loc[:, 'SK_ID_CURR'].values.tolist()

    preprocessed_features_names = preprocessor.get_feature_names_out()
    preprocessed_features_names = [re.sub('num_preprocessor__|encoder__','',s) for s in preprocessed_features_names]

    def test_preprocessing(self):
        for id in ['100001', '100141']:
            """id 100001 has a NaN in categorical column OCCUPATION_TYPE, id 100141 has a NaN in numerical column EXT_SOURCE_3"""
            sample = self.df.loc[self.df['SK_ID_CURR']==int(id), ~self.df.columns.isin(['SK_ID_CURR'])]
            preprocessed_sample = pd.DataFrame(self.preprocessor.transform(sample), columns=self.preprocessed_features_names)
            assert(preprocessed_sample.isnull().values.any()==False)

    def test_predict(self):
        id = '100005'
        sample = self.df.loc[self.df['SK_ID_CURR']==int(id), ~self.df.columns.isin(['SK_ID_CURR'])]
        preprocessed_sample = pd.DataFrame(self.preprocessor.transform(sample), columns=self.preprocessed_features_names)
        pred = self.classifier.predict(preprocessed_sample)[0].astype(float)
        pred_proba = self.classifier.predict_proba(preprocessed_sample)[0][0]
        results = {'prediction': pred, 'probability': pred_proba}
        assert(((results['prediction']==0) | (results['prediction']==1)) & ((results['prediction']>=0) & (results['prediction']<=1)))