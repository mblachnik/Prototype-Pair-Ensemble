from ppelib.classifiers import RandomOracle, RandomOracle_Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

dataDir = 'D:\\Projects\\DataMining\\Data\\CSV_Experiments\\Above_1k\\'
datasets = [

    # "Agrawal1",
    # "Stagger1", #100% dokładności
    # "BayesianNetworkGenerator_spambase",
    # "BNG_sonar",
    # (dataDirLarge,"codrnaNorm"),
    # (dataDirLarge,"electricity-normalized"),
    # (dataDirLarge,"covtype"),
    # (dataDirLarge,"php89ntbG"),

    #(dataDir, "banana"),
    #(dataDir, "coil2000"),
    #(dataDir, "magic"),
    #(dataDir, "phoneme"),
    #(dataDir, "ring"),
    (dataDir, "spambase"),
    #(dataDir, "twonorm"),

    # "shuttle2"
]

for dirName,fName in datasets:
    df = pd.read_csv(dirName + fName + "\\" + fName + ".dat", sep=";")
    X = df[[column for column in df.columns if column not in ["LABEL", "id"]]]
    y = np.squeeze(df[['LABEL']].values)
    print("Start prediction")
    models = [
              ("RO ", RandomOracle(base_estimator=RandomForestClassifier(n_jobs=5))),
              ("RF ",RandomForestClassifier(n_jobs=5)),
              ("VRO",RandomOracle_Classifier(RandomOracle(base_estimator=RandomForestClassifier(n_jobs=8, n_estimators=50)), n_estimators=20, n_jobs=1)),
               ]

    for n,m in models:
        #m = RandomOracle_Classifier(RandomOracle(base_estimator=RandomForestClassifier()), n_estimators=20)
        acc = cross_val_score(m, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=5))
        print(f"Model: {n}, Accuracy: {acc.mean() * 100:.4f}% +- {acc.std() * 100:.4f}%")
