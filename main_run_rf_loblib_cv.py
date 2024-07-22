# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:08:25 2023

@author: Marcin
"""

import pandas as pd
import numpy as np
from imblearn.under_sampling import ClusterCentroids
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import os
import ppelib.classifiers as ppe
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from joblib import Parallel, delayed
import time


from ppelib.sampler.kmeans_sampler import SimpleClusterCentroids


def parrFun(meta_columns, model, model_name, dataset, resultsDir, max_depth):
    dirName, fName = dataset

    if type(model) == ppe.PPE_Classifier:
        model.base_estimator.set_params(**{"max_depth": max_depth})
    else:
        model.set_params(**{"max_depth":max_depth})
    df = pd.read_csv(dirName + fName + ".csv", sep=";", quoting=1, quotechar='"')
    X = df[[column for column in df.columns if column not in meta_columns]]

    y = np.squeeze(df[['LABEL']].values)
    le = LabelEncoder()
    y = le.fit_transform(y)

    ohe = OneHotEncoder(sparse_output=False)
    sym_cols = (X.dtypes == "O").values
    ct = ColumnTransformer([('ohe', ohe, sym_cols)], remainder='passthrough')
    X = ct.fit_transform(X)

    cv = StratifiedKFold(random_state=1, n_splits=10, shuffle=True)

    start_time = time.time()
    score = cross_val_score(model, X, y, cv=cv, n_jobs=10)
    end_time = time.time()

    # scores[model_name] = score
    me = np.mean(score) * 100
    st = np.std(score) * 100
    print(f"{model_name}: {[me, st]}")
    res = {"model": model_name,
           "dataset": fName,
           "me": me,
           "std": st,
           "depth": max_depth,
           "scores_0": score[0],
           "scores_1": score[1],
           "scores_2": score[2],
           "scores_3": score[3],
           "scores_4": score[4],
           "scores_5": score[5],
           "scores_6": score[6],
           "scores_7": score[7],
           "scores_8": score[8],
           "scores_9": score[9],
           "train_time": start_time,
           'end_time': end_time
           }
    # if type(model) == ppe.PPE_Classifier:
    #     res["regions"] = model.region_stats.shape[0]
    tmp = pd.DataFrame([res])
    print(f"Finished calculations for {dataset} using {model_name} max_depth={max_depth} after {start_time -end_time} [s]")
    tmp.to_csv(resultsDir+ os.sep + model_name + "_" + fName + "_" + str(max_depth) + ".csv")
    return res

def gen_params(datasets,meta_columns, resultsDir):
    params = []
    for dataset in datasets:
        for max_depth in [2,3,4,5,6,7,8,9]:
            for modelName, model in models:

                params.append((meta_columns, clone(model), modelName, dataset, resultsDir, max_depth))
    return params

if __name__ == '__main__':
    parallel = True
    dataDir = 'D:\\mblachnik\\datasets\\datasets_ppe_tree\\'
    resultsDir = "Data\\tmp_results"
    #protos = 15
    script_n_jobs = 3

    base_estimator = DecisionTreeClassifier()
    # base_estimator = RandomForestClassifier(n_estimators=100, n_jobs=10)
    # base_estimator = Pipeline([("Scale",StandardScaler()),("PCA", PCA(n_components=None)), ("RF", base_estimator)])
    models = [
         ("PPE2", ppe.PPE_Classifier(type="ppe2",
                                    base_estimator=clone(base_estimator),
                                    # proto_selection=ClusterCentroids(sampling_strategy={-1:5,1:5}),
                                    proto_selection=ClusterCentroids(sampling_strategy={0: 10, 1: 10}),
                                    unbalanced_rate=0.2,
                                    minimum_regions=2,
                                    min_support=400,
                                    n_jobs=6)),

        ("Tree", clone(base_estimator)),
        ("RF", RandomForestClassifier())
        ]

    datasets = [

                    # ,
                # "Stagger1", #100% dokładności
                # "BayesianNetworkGenerator_spambase",
                #"BNG_sonar",
        #(dataDir,"codrnaNorm"),
        #(dataDir,"electricity-normalized"),
        #(dataDir,"covtype"),
        (dataDir,"phpvcoG8S"),
        (dataDir,"Agrawal1"),
        (dataDir,"shuttle2"),
        #(dataDir, "banana"),
        (dataDir, "coil2000"), #Mocno niezbalansowany
        (dataDir, "magic"),
        #(dataDir, "phoneme"),
        #(dataDir, "ring"),
        (dataDir, "spambase"),
        (dataDir, "twonorm"),
        (dataDir, "titanic"),

                ]

    meta_columns = ["LABEL","id"]

    params = gen_params(datasets, meta_columns, resultsDir)

    if parallel:
        with Parallel(n_jobs=script_n_jobs) as parallel:
            start_time = time.time()
            all_res = parallel(delayed(parrFun)(*param) for param in params)
            print("--- %s seconds ---" % (time.time() - start_time))

    else:
        all_res = []
        for param in params:
            print(param)
            out = parrFun(*param)
            all_res.append(out)


    df = pd.DataFrame(all_res)
    df.to_excel("Data\\wyniki_tree_2.xlsx")

    # adf = df.groupby(by=["dataset","model"]).agg(["mean","std"])
    # adf.to_excel("Data\\wyniki_tree_2_agg.xlsx")