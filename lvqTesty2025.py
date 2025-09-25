import json
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklvq import GLVQ

from ppelib.validationSolver import CustomSteepestGradientDescent
from ppelib.new_metric import mlflow_scoring_wrapper

mlflow.set_tracking_uri("http://192.168.10.40:5000")


def login():
    if os.environ.get("MLFLOW_TRACKING_USERNAME",None) is None or os.environ.get("MLFLOW_TRACKING_PASSWORD",None) is None:
        user = input("USER: ")
        password = input("PASSWORD: ")
        os.environ["MLFLOW_TRACKING_USERNAME"] = user.strip()
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password.strip()

login()


folder = "Y:/Datasets/Datasets/KeelNorm"
resFolder = "Data/Results/"
files = ["banana"]#,"electricity-normalized"]

def gen_solver_params(max_runs:list, step_size:list, batches:list, patiences:list, val_splits:list, early_stoping:list = [False]):
    res = []
    for m in max_runs:
        for s in step_size:
            for b in batches:
                for v in val_splits:
                    for p in patiences:
                        for e in early_stoping:
                            res.append({
                                "max_runs": m,
                                "step_size": s,
                                "batch_size": b,
                                "val_split": v,
                                "patience": p,
                                "early_stopping": e,
                            })
    return res

N_JOBS = 32
prototypes_n_per_class = [3, 5, 9, 15]
max_runs = [50, 500, 1000]
steps = [0.01, 0.001, 0.005]
batches = [1,8,64,256,1024,0]
val_splits = [0.1]
patiences = [10,20]
early_stoping = [True,False]
solver_params = gen_solver_params(max_runs, steps, batches, patiences, val_splits, early_stoping)

for file in files:
    mlflow.set_experiment(f"LVQObjectiveEarlyStoping2{file}") 
    try:
        data = pd.read_csv(f"{folder}\\{file}\\{file}.dat", sep=";")
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

    ohe = LabelEncoder()
    cols = [col for col in data.columns if col not in ["LABEL", "id"]]
    X = data.loc[:, cols].values
    y = data.loc[:, "LABEL"].values
    y = ohe.fit_transform(y)

    with mlflow.start_run(run_name=f"GLVQ_{file}"):
        mlflow.set_tag("dataset", file)

        model_main = GLVQ(
            distance_type="squared-euclidean",
            activation_type="sigmoid",
            activation_params={"beta": 1},
            solver_type=CustomSteepestGradientDescent,
            solver_params={"max_runs": 20, "step_size": 0.1},
            random_state=42,
            prototype_n_per_class=5,
        )

        model = GridSearchCV(
            model_main,
            {
                "prototype_n_per_class": prototypes_n_per_class,
                "solver_params": solver_params
            },
            cv=StratifiedKFold(n_splits=5),
            scoring=mlflow_scoring_wrapper,
            n_jobs=N_JOBS,
            verbose=3
        )

        model.fit(X, y)

        test_df = pd.DataFrame(model.cv_results_)
        out_path = f"{resFolder}\\{file}_grid_res.csv"
        test_df.to_csv(out_path, index=False, sep=";")

        try:
            dataset_info = {
                "name": "covtype",
                "path": "Y:/Datasets/Datasets/KeelNorm/covtype",
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "feature_names": cols,
                "classes": list(np.unique(y))
            }
            with open("dataset_info.json", "w") as f:
                json.dump(dataset_info, f)

            mlflow.log_artifact("dataset_info.json")
            os.remove("dataset_info.json")
        except:pass