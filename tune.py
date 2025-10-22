import dagshub
dagshub.init(repo_owner='abdelrahman-ab',
             repo_name='MLops_lab3_Task',
             mlflow=True)

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train.drop("class", axis=1)
y_train = train["class"]
X_test = test.drop("class", axis=1)
y_test = test["class"]

n_estimators_list = [50, 100, 150]
max_depth_list = [5, 10, 15]

with mlflow.start_run(run_name="tuning_experiment"):
    for n in n_estimators_list:
        for d in max_depth_list:
            with mlflow.start_run(run_name=f"n={n}_d={d}", nested=True):
                model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", d)
                mlflow.log_metric("accuracy", acc)

print("Check DagsHub Experiments tab for the results.")
