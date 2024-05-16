# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:08:25 2023

@author: Marcin
"""

import pandas as pd
import numpy as np
from imblearn.under_sampling import ClusterCentroids
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import os
from ppelib import ppe
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from joblib import Parallel, delayed
import time


from ppelib.sampler.kmeans_sampler import SimpleClusterCentroids


def parrFun(meta_columns, model, modelName, dataset, fileId, resultsDir):
    dataDir,dataset = dataset
    traFile = f"{dataset}-10-{fileId}tra.dat.csv"
    tstFile = f"{dataset}-10-{fileId}tst.dat.csv"
    dfTr = pd.read_csv(dataDir + os.sep + dataset + os.sep + traFile, sep=";")
    dfTe = pd.read_csv(dataDir + os.sep + dataset + os.sep + tstFile, sep=";")
    if any(["id" in col for col in dfTr.columns]):
        dfTr = dfTr.set_index("id")
        dfTe = dfTe.set_index("id")
    X = dfTr[[col for col in dfTr.columns if col not in meta_columns]].values
    y = np.squeeze(dfTr[['LABEL']].values)
    oh = LabelEncoder()
    y = oh.fit_transform(y)
    XTe = dfTe[[col for col in dfTe.columns if col not in meta_columns]].values
    yTe = np.squeeze(dfTe[['LABEL']].values)
    yTe = oh.transform(yTe)
    res = {"dataset": dataset}
    print(f"==> Starting calculations for {dataset} using {modelName} CV_{fileId}")
    m = clone(model)
    fit_start_time = time.time()
    m.fit(X, y)
    fit_end_time = time.time()
    yp = m.predict(XTe)
    predict_end_time = time.time()
    print(f"Finished calculations for {dataset} using {modelName} CV_{fileId} after {fit_start_time- predict_end_time} [s]")
    acc = np.mean(yTe == yp)
    bacc = balanced_accuracy_score(yTe,yp)
    res["acc"] = acc
    res["bacc"] = bacc
    res["model"] = modelName
    #print(f"{modelName} ACC={acc}")
    res["train_time"] = fit_end_time - fit_start_time
    res["predict_time"] = predict_end_time - fit_end_time
    # id = list(m.fitted_base_models_.keys())[0]
    # res["C"] = m.fitted_base_models_[id].best_params_["C"]
    # res["gamma"] = m.fitted_base_models_[id].best_params_["gamma"]
    if type(model)==ppe.PPE_Classifier:
        res["regions"] = len(m.regions_) #.shape[0]
    else:
        res["regions"] = 1
    tmp = pd.DataFrame([res])
    tmp.to_csv(resultsDir+ os.sep + modelName + "_" + tstFile + ".csv")
    return res

def gen_params(datasets,meta_columns, resultsDir):
    params = []
    for dataset in datasets:
        for fileId in range(1, 11):
            for modelName, model in models:
                params.append((meta_columns, model, modelName, dataset, fileId, resultsDir))
    return params

if __name__ == '__main__':
    parallel = True
    dataDir = r'D:\mblachnik\datasets\Datasets\KeelNormCV'
    dataDirLarge = "D:\\mblachnik\\datasets\\large"
    resultsDir = "Data\\tmp_results"
    protos = 15
    script_n_jobs = 30

    base_estimator = SVC(C=1, gamma='auto', cache_size=200)
    base_estimator = GridSearchCV(estimator=SVC(),
                                  param_grid={'C': [0.01, 1, 100],
                                              'gamma': [0.01, 0.1, 1, 10]},
                                  n_jobs=5,
                                  )
    # base_estimator = RandomForestClassifier(n_estimators=100, n_jobs=10)
    # base_estimator = Pipeline([("Scale",StandardScaler()),("PCA", PCA(n_components=None)), ("RF", base_estimator)])
    models = [
        # ("PE", ppe.PPE_Classifier(base_estimator=base_estimator,
        #                           type="pe",proto_selection={0:protos, 1:protos}, min_support=400, unbalanced_rate=0.05)),
        # ("PPE", ppe.PPE_Classifier(base_estimator=base_estimator,
        #                            type="ppe", proto_selection={0: protos, 1: protos}, min_support=400, unbalanced_rate=0.05)),
        ("PE",  ppe.PPE_Classifier(base_estimator=base_estimator,
                                   type="pe",
                                   proto_selection=SimpleClusterCentroids(n_clusters=20),
                                   min_support=100,
                                   unbalanced_rate=0.001,
                                   minimum_regions=2,
                                   n_jobs=5
                                   )),
        # ("PPE3", ppe.PPE_Classifier(base_estimator=base_estimator,
        #                             type="ppe3",
        #                             proto_selection=ClusterCentroids(estimator=KMeans(random_state=0, n_init=10),
        #                                 sampling_strategy={0: 25, 1: 25}),
        #                             min_support=100,
        #                             unbalanced_rate=0.01,
        #                             minimum_regions=2,
        #                             prune_regions=False,
        #                             n_jobs=5)),
        ("PPE", ppe.PPE_Classifier(base_estimator=base_estimator,
                                   type="ppe2",
                                   proto_selection=ClusterCentroids(estimator=KMeans(random_state=0, n_init=10),
                                        sampling_strategy={0: 10, 1: 10}),
                                   min_support=1000,
                                   unbalanced_rate=0.01,
                                   minimum_regions=2,
                                   n_jobs=5)),
        # ("RF", RandomForestClassifier()),
        # ("PCA+RF", base_estimator),
        # ("EPPE",ppe.EPPE_Classifier(ppe_estimator=
        #                      ppe.PPE_Classifier(base_estimator=RandomForestClassifier(n_estimators=10),
        #                                         proto_selection={0:protos, 1:protos}), #Warning: Here it must be class 0,1 instead of -1,1 becouse VotingEnsemble use onehot label encodings which converts output labels into values [0,1]
        #                      n_estimators=10)),

        #
        #("BASE",base_estimator),
        ]

    datasets = [

                    # "Agrawal1",
                # "Stagger1", #100% dokładności
                # "BayesianNetworkGenerator_spambase",
                #"BNG_sonar",
        (dataDirLarge,"codrnaNorm"),
        (dataDirLarge,"electricity-normalized"),
        (dataDirLarge,"covtype"),
        (dataDirLarge,"php89ntbG"),


        # (dataDir, "banana"),
        # #(dataDir, "coil2000"), #Mocno niezbalansowany
        # (dataDir, "magic"),
        # (dataDir, "phoneme"),
        # (dataDir, "ring"),
        # (dataDir, "spambase"),
        # (dataDir, "twonorm"),


                 #"shuttle2"
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
    df.to_excel("Data\\wyniki_svm_2.xlsx")

    adf = df.groupby(by=["dataset","model"]).agg(["mean","std"])
    adf.to_excel("Data\\wyniki_svm_2_agg.xlsx")