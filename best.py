import logging
import pandas as pd
import numpy as np
import json
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, kmeans_model=None):
        self.kmeans_model = kmeans_model

    def fit(self, X, y=None):
        if self.kmeans_model is None:
            self.kmeans_model = KMeans(n_clusters=10, random_state=123)
            self.kmeans_model.fit(X[['lat', 'lon']])
        return self
   
    def transform(self, X):
        X_ = X.copy()
        X_['num_reviews_log'] = np.log1p(X_['num_reviews'])
        X_['rating_sq'] = X_['rating'] ** 2
        X_['rating_bin'] = pd.cut(X_['rating'], bins=[-np.inf, 3, 4.5, np.inf], labels=[0,1,2]).astype(int)
        X_['guests_sq'] = X_['guests'] ** 2
        X_['rating_x_reviews'] = X_['rating'] * X_['num_reviews_log']
        X_['guests_x_rating'] = X_['guests'] * X_['rating']
        X_['location_cluster'] = self.kmeans_model.predict(X[['lat', 'lon']])
        return X_


def extract_facilities(df, top_k=20):
    df['facilities_list'] = df['facilities'].fillna("").apply(lambda x: x.split())
    all_facilities = [fac for sublist in df['facilities_list'] for fac in sublist]
    most_common = [item for item, _ in Counter(all_facilities).most_common(top_k)]
    
    for facility in most_common:
        df[f'has_{facility}'] = df['facilities_list'].apply(lambda lst: int(facility in lst))
    
    return df.drop(columns=['facilities_list'])

def baseline():
    logging.info("Reading train and test files")
    train = pd.read_json("train.json", orient='records')
    test = pd.read_json("test.json", orient='records')
    train = extract_facilities(train)
    test = extract_facilities(test)
    
    for df in [train, test]:
        df["num_facilities"] = df["facilities"].str.split().apply(len)
    facility_features = [col for col in train.columns if col.startswith("has_")]
    train, valid = train_test_split(train, test_size=1/3, random_state=123)

    numerical_features = ["lat", "lon", "rooms", "beds", "bathrooms","guests", "num_reviews", 
                          "min_nights", "rating", "num_facilities","num_reviews_log", "rating_sq",
                          "rating_bin", "guests_sq","rating_x_reviews", "guests_x_rating"] + facility_features
    categorical_features = ["room_type", "listing_type", "cancellation", "location_cluster"]
    

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='mean')),
            ]), numerical_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("encoder", OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_features),
        ]
    )
    label = 'revenue'
    models = {
        "mean": Pipeline([("feature_engineering", FeatureEngineer()),
                          ("preprocess", preprocess), ("regressor", DummyRegressor())]),
        "ridge": Pipeline([("feature_engineering", FeatureEngineer()),
                ("preprocess", preprocess), ("regressor", Ridge(alpha=1, random_state=123))]),
        
        "random_forest": Pipeline([("feature_engineering", FeatureEngineer()),
            ("preprocess", preprocess),("regressor", RandomForestRegressor(random_state=123, n_jobs=-1, n_estimators=300, max_depth=15))]),
        
        "hist_gradient_boosting": Pipeline([ ("feature_engineering", FeatureEngineer()),
            ("preprocess", preprocess),("regressor", HistGradientBoostingRegressor(random_state=123, learning_rate=0.05, max_iter= 1000))]),
        "xgboost": Pipeline([("feature_engineering", FeatureEngineer()),
            ("preprocess", preprocess),("regressor", XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=4,subsample=0.8,colsample_bytree=0.7, reg_lambda = 1.0, random_state=123, n_jobs=-1))])}

    for model_name, model in models.items():
        logging.info(f"Fitting model {model_name}")
        model.fit(train.drop([label], axis=1), np.log1p(train[label].values))
        for split_name, split in [("train     ", train),
                                  ("valid     ", valid)]:
            pred = np.expm1(model.predict(split.drop([label], axis=1)))
            mae = mean_absolute_error(split[label], pred)
            logging.info(f"{model_name} {split_name} {mae:.3f}")                    
        
    pred_test = np.expm1(models["xgboost"].predict(test))
    test[label] = pred_test
    predicted = test[['revenue']].to_dict(orient='records')
    with zipfile.ZipFile("predicted.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        # Write to a file inside the ZIP
        zipf.writestr("predicted.json", json.dumps(predicted, indent=2))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    baseline()

