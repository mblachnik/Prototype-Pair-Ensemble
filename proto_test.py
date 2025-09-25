import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import tempfile

from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklvq import GLVQ
from ppelib.validationSolver import CustomSteepestGradientDescent
from imblearn.under_sampling import ClusterCentroids
from utils.mlflow_utils import save_fig_as_artefact

def login():
    if os.environ.get("MLFLOW_TRACKING_USERNAME",None) is None or os.environ.get("MLFLOW_TRACKING_PASSWORD",None) is None:
        user = input("USER: ")
        password = input("PASSWORD: ")
        os.environ["MLFLOW_TRACKING_USERNAME"] = user.strip()
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password.strip()

def get_fig(pX, py, title, X_data=None,y_data=None, cols=None,x_axis=0,y_axis=1):
    fig, ax = plt.subplots(figsize=(6, 6))
    pX0 = pX[py == 0]
    pX1 = pX[py == 1]

    print(pX)
    ax.set_title(f"{title}")
    if X_data is not None and y_data is not None:
        X0 = X_data[y_data == 0]
        X1 = X_data[y_data == 1]
        ax.scatter(X0[:,x_axis], X0[:,y_axis], marker="o", color='green', s=8, label="Class 0")
        ax.scatter(X1[:,x_axis], X1[:,y_axis], marker="*", color='blue', s=8, label="Class 1")
    ax.scatter(pX0[:,x_axis], pX0[:,y_axis], marker="o", color='red', s=80, label="Proto Class 0")
    ax.scatter(pX1[:,x_axis], pX1[:,y_axis], marker="*", color='red', s=80, label="Proto Class 1")
    if cols is not None:
        ax.set_xlabel(cols[x_axis])
        ax.set_ylabel(cols[y_axis])
    ax.legend()
    ax.grid(True)
    return fig

def get_fig_all(pX, py, names, title, X_data=None,y_data=None, cols=None,x_axis=0,y_axis=1):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"{title}")
    colors = {0:"red",1:"yellow",2:"black"}
    if X_data is not None and y_data is not None:
        X0 = X_data[y_data == 0]
        X1 = X_data[y_data == 1]
        ax.scatter(X0[:,x_axis], X0[:,y_axis], marker="o", color='green', s=8, label="Class 0")
        ax.scatter(X1[:,x_axis], X1[:,y_axis], marker="*", color='blue', s=8, label="Class 1")
    for i, n in enumerate(names): 
        pX0 = pX[i][py[i] == 0]
        pX1 = pX[i][py[i] == 1]
        ax.scatter(pX0[:,x_axis], pX0[:,y_axis], marker="o", color=colors.get(i,"white"), s=80, label=f"Proto Class 0 [{n}]")
        ax.scatter(pX1[:,x_axis], pX1[:,y_axis], marker="*", color=colors.get(i,"white"), s=80, label=f"Proto Class 1 [{n}]")
    if cols is not None:
        ax.set_xlabel(cols[x_axis])
        ax.set_ylabel(cols[y_axis])
    ax.legend()
    ax.grid(True)
    return fig


#Params
N_PROTOTYPE = 5
MAX_RUNS = 5
BATCHES = 0
STEP_SIZE = 0.1
RAND = 42
EXP_NAME = "Prototype_test"
TEST_NAME = "TEST_RUN"
knn = ClusterCentroids(sampling_strategy={0: N_PROTOTYPE, 1: N_PROTOTYPE},random_state=RAND)

lvq = GLVQ(
            distance_type="squared-euclidean",
            activation_type="sigmoid",
            activation_params={"beta": 1},
            solver_type="steepest-gradient-descent",
            solver_params={"max_runs": MAX_RUNS, "step_size": STEP_SIZE},
            random_state=RAND,
            prototype_n_per_class=N_PROTOTYPE,
        )
lvq_early_stoping = GLVQ(
            distance_type="squared-euclidean",
            activation_type="sigmoid",
            activation_params={"beta": 1},
            solver_type=CustomSteepestGradientDescent,
            solver_params={"max_runs": MAX_RUNS, "step_size": STEP_SIZE, 'early_stopping':True},
            random_state=RAND,
            prototype_n_per_class=N_PROTOTYPE
        )

folder = "Y:/Datasets/Datasets/KeelNorm"
file = "banana"
data = pd.read_csv(f"{folder}\\{file}\\{file}.dat", sep=";")

ohe = LabelEncoder()
cols = [col for col in data.columns if col not in ["LABEL", "id"]]
X = data.loc[:, cols].values
y = data.loc[:, "LABEL"].values
y = ohe.fit_transform(y)

tmp_dir = tempfile.mkdtemp()

os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "passwor"
login()
mlflow.set_tracking_uri("http://192.168.10.40:5000")
mlflow.set_experiment(EXP_NAME)

with mlflow.start_run(run_name=TEST_NAME):
    mlflow.set_tag("n_prototype", N_PROTOTYPE)
    mlflow.set_tag("lvq_max_run", MAX_RUNS)
    mlflow.set_tag("lvq_batches", BATCHES)
    mlflow.set_tag("lvq_step_size", STEP_SIZE)

    #KNN
    X_knn, y_knn = knn.fit_resample(X, y)
    knn_fig = get_fig(X_knn, y_knn, "ClusterCentroids", X,y, cols)
    save_fig_as_artefact(knn_fig, "KNN",tmp_dir)

    #LVQ
    lvq.fit(X, y)
    X_lvq, y_lvq = lvq.prototypes_, lvq.prototypes_labels_
    lvq_fig = get_fig(X_lvq, y_lvq, "LVQ", X,y, cols)
    save_fig_as_artefact(lvq_fig, "LVQ",tmp_dir)
    #LVQ EARLYSTOPING
    lvq_early_stoping.fit(X, y)
    X_lvq_es, y_lvq_es = lvq_early_stoping.prototypes_, lvq_early_stoping.prototypes_labels_
    lvq_es_fig = get_fig(X_lvq_es, y_lvq_es, "LVQ EARLY STOPING", X,y, cols)
    save_fig_as_artefact(lvq_es_fig, "LVQ_ES",tmp_dir)
    all_fig = get_fig_all([X_knn,X_lvq,X_lvq_es],[y_knn,y_lvq,y_lvq_es],["knn","lvq","lvq_es"],"Prototypes", X,y, cols)
    save_fig_as_artefact(all_fig, "ALL",tmp_dir)