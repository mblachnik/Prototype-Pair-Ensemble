# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:08:25 2023

@author: Marcin
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os
from ppelib import ppe
from sklearn.base import clone
import multiprocessing


dataDir = r'D:\mblachnik\datasets\Datasets\KeelNormCV'
protos = 15
models = [
    ("PE", ppe.PPE_Classifier(base_estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1),
                              type="pe",proto_selection={0:protos, 1:protos}, min_support=400)),
    ("PPE",ppe.PPE_Classifier(base_estimator=RandomForestClassifier(n_estimators=100, n_jobs=--1),
                              type="ppe",proto_selection={0:protos, 1:protos}, min_support=400)),
    ("EPPE",ppe.EPPE_Classifier(ppe_estimator=
                         ppe.PPE_Classifier(base_estimator=RandomForestClassifier(n_estimators=10),
                                            proto_selection={0:protos, 1:protos}), #Warning: Here it must be class 0,1 instead of -1,1 becouse VotingEnsemble use onehot label encodings which converts output labels into values [0,1]
                         n_estimators=10)),
     ("RF",RandomForestClassifier(n_estimators=100, n_jobs=-1))]

datasets = ["banana",
            #"coil2000",
    "magic",
    "phoneme",
    "ring",
    #"shuttle2",
    "spambase",
    "titanic",
    "twonorm"]
meta_columns = ["LABEL","id"]
all_res = []
for dataset in datasets:
    print("==================================================")
    print(dataset)
    #with multiprocessing.Pool(50) as pool:
    for fileId in range(1,10):
        traFile = f"{dataset}-10-{fileId}tra.dat.csv"
        tstFile = f"{dataset}-10-{fileId}tst.dat.csv"
        dfTr = pd.read_csv(dataDir + os.sep + dataset + os.sep + traFile,sep=";")
        dfTe = pd.read_csv(dataDir + os.sep + dataset + os.sep + tstFile,sep=";")
        dfTr = dfTr.set_index("id")
        dfTe = dfTe.set_index("id")
        X = dfTr[[col for col in dfTr.columns if col not in meta_columns]].values
        y = np.squeeze(dfTr[['LABEL']].values)
        oh = LabelEncoder()
        y = oh.fit_transform(y)
        XTe = dfTe[[col for col in dfTe.columns if col not in meta_columns]].values
        yTe = np.squeeze(dfTe[['LABEL']].values)
        yTe = oh.transform(yTe)
        res = {"dataset":dataset}
        print(f"CV_{fileId}")
        for modelName,model in models:
            m = clone(model)
            m.fit(X, y)
            yp = m.predict(XTe)
            acc = np.mean(yTe == yp)
            res[modelName] = acc
            print(f"{modelName} ACC={acc}")
        all_res.append(res)

df = pd.DataFrame(all_res)
df.to_excel("Data\\wyniki.xlsx")
adf = df.groupby(by="dataset").agg(["mean", "std"])
adf.to_excel("Data\\wyniki_agg.xlsx")
