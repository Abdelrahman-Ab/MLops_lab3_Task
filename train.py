import dagshub
dagshub.init(repo_owner='abdelrahman-ab',
             repo_name='MLops_lab3_Task',
             mlflow=True)

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train.drop("class", axis=1)
y_train = train["class"]
X_test = test.drop("class", axis=1)
y_test = test["class"]

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", acc)

    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")

print(f"Model trained and saved! Accuracy: {acc:.4f}")
