import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from feast import FeatureStore
from feast.data_source import PushMode
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, root_mean_squared_error
from flaml import AutoML
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import CloudWorkspace
import pickle
import warnings
warnings.filterwarnings('ignore')


def fetch_historical_features_entity_df(store: FeatureStore, entity_df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Takes an entity_df and returns features from Feast."""
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=features,
    ).to_df()

    return training_df



def pipeline(X, model):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    one_hot_features = X[['brand', 'fuel_type', 'transmission']].columns.tolist()
    target_encode_features = X[['model']].columns.tolist()
    ordinal_encode_features = X[['condition']].columns.tolist()

    condition_order = [['Used', 'Like New', 'New']]

    numeric_preprocess = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    one_hot_preprocess = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    target_encode_preprocess = Pipeline(steps=[
        ("target_encode", TargetEncoder()),
    ])

    ordinal_encode_preprocess = Pipeline(steps=[
        ("ordinal", OrdinalEncoder(categories=condition_order)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_preprocess, numeric_features),
        ("onehot", one_hot_preprocess, one_hot_features),
        ("target", target_encode_preprocess, target_encode_features),
        ("ordinal", ordinal_encode_preprocess, ordinal_encode_features),
    ])


    lr = model

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', lr),
    ])

    return pipeline

def run_model(df):
    X = df.drop(columns=["price"])
    y = df["price"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = AutoML()

    # make pipeline
    pipe = pipeline(X, model)

    pipe.fit(X_train, y_train, model__task='regression', model__time_budget=60, model__metric='rmse', model__log_file_name='automl_all_features_log.txt')

    return pipe


def run_demo():
    store = FeatureStore(repo_path=".")
    subprocess.run(["feast", "apply"])

    # Example raw data before cleaning or feature fetching
    raw_df = pd.read_csv("data/car_price_prediction_.csv")

    subprocess.run(["git", "init"])
    subprocess.run(["dvc", "init", "-f"])
    subprocess.run(["dvc", "add", "data/car_price_prediction_.csv"])
    subprocess.run(["git", "add", "data\.gitignore", "data/car_price_prediction_.csv.dvc"])
    subprocess.run(["git", "commit", "-m'Original data'"])
    subprocess.run(["git", "tag", "original_data"])
    subprocess.run(["dvc", "commit"])
    subprocess.run(["git", "log"])



    #add event timestamp
    dates = [
        pd.Timestamp("2025-10-24"),
        pd.Timestamp("2025-10-25"),
    ]

    # Split indices into 2 roughly equal parts
    indices = np.array_split(raw_df.index, 2)
    for i, idx in enumerate(indices):
        raw_df.loc[idx, "event_timestamp"] = dates[i]
    
    # ensure timestamp is in datetime format
    raw_df["event_timestamp"] = pd.to_datetime(raw_df["event_timestamp"], utc=True)

    # save data as parquet for feast file source
    raw_df.to_parquet("data/car_data.parquet", index=False)

    # prepare feature request for 10-21
    request = raw_df[raw_df['event_timestamp'] == pd.to_datetime(pd.Timestamp("2025-10-24"), utc = True)][['car_id','event_timestamp']]

    # prepare feature request for version 1
    features=[
            "car_stats:brand",
            "car_stats:year",
            "car_stats:engine_size",
            "car_stats:fuel_type",
            "car_stats:transmission",
            "car_stats:mileage",
            "car_stats:condition",
            "car_stats:price",
            "car_stats:model",
        ]

    # get data from feast
    df = fetch_historical_features_entity_df(store, request, features)

    df = df.drop(columns = ['car_id', 'event_timestamp'])

    df.to_csv("data/car_price_prediction_.csv", index=False) #replace original (will be okay, i am versioning with dvc/git) :)
    subprocess.run(["dvc", "add", "data/car_price_prediction_.csv"])
    subprocess.run(["git", "add", "data/car_price_prediction_.csv.dvc"])
    subprocess.run(["git", "commit", "-m'Unchanged'"])
    subprocess.run(["git", "tag", "unchanged"])
    subprocess.run(["dvc", "commit"])

    schema = DataDefinition(
        numerical_columns=['year', 'engine_size', 'mileage'],
    )

    eval_data_original = Dataset.from_pandas(
        pd.DataFrame(df),
        data_definition=schema
    )

    pipe = run_model(df)

    results = {}
    print(f"Best estimator: {pipe.named_steps['model'].best_estimator}")  
    results['best_model_top_3_features'] = pipe.named_steps['model'].best_estimator
    print(f"Best configuration: {pipe.named_steps['model'].best_config}")  
    results['best_config_top_3_features'] = pipe.named_steps['model'].best_config
    print(f"Best RMSE on validation data: {pipe.named_steps['model'].best_loss}")

    pickle.dump(pipe, open('model_no_drift.pkl', 'wb'))


    # fetch data again and simulate data drift by modifying the 'mileage' and 'year' features
    # get data from feast
    drift_df = fetch_historical_features_entity_df(store, request, features)
    drift_df = drift_df.drop(columns = ['car_id', 'event_timestamp'])
    drift_df['mileage'] = drift_df['mileage'] * 1.5  # increase mileage by 50%
    drift_df['year'] = drift_df['year'] - 10  # decrease year by 10

    drift_df.to_csv("data/car_price_prediction_.csv", index=False) #replace original (will be okay, i am versioning with dvc/git) :)
    subprocess.run(["dvc", "add", "data/car_price_prediction_.csv"])
    subprocess.run(["git", "add", "data/car_price_prediction_.csv.dvc"])
    subprocess.run(["git", "commit", "-m'modified mileage to simulate data drift'"])
    subprocess.run(["git", "tag", "modified"])
    subprocess.run(["dvc", "commit"])

    eval_data_drift = Dataset.from_pandas(
        pd.DataFrame(drift_df),
        data_definition=schema
    )

    report = Report([
        DataDriftPreset()
    ])

    drift_pipe = run_model(drift_df)
    print(f"Best estimator: {drift_pipe.named_steps['model'].best_estimator}")  
    results['best_model_top_3_features'] = drift_pipe.named_steps['model'].best_estimator
    print(f"Best configuration: {drift_pipe.named_steps['model'].best_config}")  
    results['best_config_top_3_features'] = drift_pipe.named_steps['model'].best_config
    print(f"Best RMSE on validation data: {drift_pipe.named_steps['model'].best_loss}")

    ws = CloudWorkspace(token="dG9rbgH8tIyC2S5GmaVh0iN2lpcW5ZFpSAN3kBu6YrQdMHpNcQBQ8J+DU1MVMH0o1bH10JejdX4FdYR1LnAQLs8o3XKQubLZ9qH7guXdY90lDY8+ps8Tktu8THfI0SYwNLCneoq+83OVzvbY/0J7mRvDZl9SRNdsJZ++", url="https://app.evidently.cloud")
    project_id = "019ab7c4-fde4-7695-aa22-616bfdf872cc"
    project = ws.get_project(project_id)


    my_eval = report.run(eval_data_original, eval_data_drift)

    ws.add_run(project_id, my_eval, include_data=False)



    pickle.dump(drift_pipe, open('model_drift.pkl', 'wb'))



    # Rollback to original data
    subprocess.run(["git", "checkout", "original_data"])
    subprocess.run(["dvc", "checkout"])
    print("Rolled back to the original data")

    print("\n--- Run feast teardown ---")
    subprocess.run(["feast", "teardown"])

if __name__ == "__main__":
    run_demo()
