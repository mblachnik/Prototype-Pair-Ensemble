# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:08:25 2023

@author: Marcin
"""

import pandas as pd
import numpy as np
from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import os
from ppelib import ppe
from sklearn.base import clone
from sklearn.svm import SVC
from joblib import Parallel, delayed
import time

from ppelib.sampler.kmeans_sampler import SimpleClusterCentroids


def parrFun(meta_columns, model, modelName, dataset, fileId):
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
    print(f"CV_{fileId}")
    m = clone(model)
    fit_start_time = time.time()
    m.fit(X, y)
    fit_end_time = time.time()
    yp = m.predict(XTe)
    predict_end_time = time.time()
    acc = np.mean(yTe == yp)
    res["accuracy"] = acc
    res["model"] = modelName
    #print(f"{modelName} ACC={acc}")
    res["train_time"] = fit_end_time - fit_start_time
    res["predict_time"] = predict_end_time - fit_end_time
    if type(model)==ppe.PPE_Classifier:
        res["regions"] = m.regions_.shape[0]
    else:
        res["regions"] = 1
    return res

def gen_params(datasets,meta_columns):
    params = []
    for dataset in datasets:
        for fileId in range(1, 11):
            for modelName, model in models:
                params.append((meta_columns, model, modelName, dataset, fileId))
    return params

if __name__ == '__main__':
    parallel = False
    dataDir = r'D:\mblachnik\datasets\Datasets\KeelNormCV'
    dataDirLarge = "D:\\mblachnik\\datasets\\large"
    protos = 15
    #base_estimator = RandomForestClassifier(n_estimators=100, n_jobs=10)
    base_estimator = SVC(C=1, gamma='auto', cache_size=200)
    models = [
        # ("PE", ppe.PPE_Classifier(base_estimator=base_estimator,
        #                           type="pe",proto_selection={0:protos, 1:protos}, min_support=400, unbalanced_rate=0.05)),
        # ("PPE", ppe.PPE_Classifier(base_estimator=base_estimator,
        #                            type="ppe", proto_selection={0: protos, 1: protos}, min_support=400, unbalanced_rate=0.05)),
        ("PE",  ppe.PPE_Classifier(base_estimator=base_estimator,
                                   type="pe",
                                   proto_selection=SimpleClusterCentroids(n_clusters=20),
                                   min_support=400,
                                   minimum_regions=2
                                   )),
        ("PPE2", ppe.PPE_Classifier(base_estimator=base_estimator,
                                    type="ppe2",
                                    proto_selection=ClusterCentroids(estimator=KMeans(random_state=0, n_init=10),
                                        sampling_strategy={0: 15, 1: 15}),
                                    min_support=400,
                                    minimum_regions=2)),
        ("PPE", ppe.PPE_Classifier(base_estimator=base_estimator,
                                   type="ppe",
                                   proto_selection=ClusterCentroids(estimator=KMeans(random_state=0, n_init=10),
                                       sampling_strategy={0: 15, 1: 15}),
                                   min_support=400,
                                   minimum_regions=2))
        # ("EPPE",ppe.EPPE_Classifier(ppe_estimator=
        #                      ppe.PPE_Classifier(base_estimator=RandomForestClassifier(n_estimators=10),
        #                                         proto_selection={0:protos, 1:protos}), #Warning: Here it must be class 0,1 instead of -1,1 becouse VotingEnsemble use onehot label encodings which converts output labels into values [0,1]
        #                      n_estimators=10)),

        #
        #("BASE",base_estimator),
        ]

    datasets = [ (dataDir,"coil2000")]
    q = [
                # "Agrawal1",
                # "Stagger1", #100% dokładności
                # "BayesianNetworkGenerator_spambase",
                #"BNG_sonar",
        # (dataDirLarge,"codrnaNorm"),
        # (dataDirLarge,"electricity-normalized"),
        # (dataDirLarge,"covtype"),
        # (dataDirLarge,"php89ntbG"),



        (dataDir,"spambase"),
        (dataDir,"banana"),
        (dataDir,"phoneme"),
        (dataDir,"ring"),
        (dataDir,"twonorm"),
        (dataDir,"coil2000"),
        (dataDir,"magic"),
                 #"shuttle2"
                ]

    meta_columns = ["LABEL","id"]

    params = gen_params(datasets,meta_columns)

    if parallel:
        with Parallel(n_jobs=40) as parallel:
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