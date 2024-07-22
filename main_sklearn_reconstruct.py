# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:08:25 2023

@author: Marcin
"""
#ToDo: https://github.com/omadson/fuzzy-c-means/blob/master/fcmeans/main.py
import pandas as pd
import numpy as np
from scipy.stats import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from ppelib.sampler.kmeans_sampler import SimpleClusterCentroids
from imblearn.under_sampling import ClusterCentroids
from sklearn.pipeline import Pipeline
from ppelib import classifiers as  ppe
from sklearn.base import clone
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score




#dataDir = 'D:\\Projects\\DataMining\\Data\\CSV_Experiments\\Above_1k\\'
#dataDirLarge = 'D:\\Projects\\DataMining\\Data\\Large - OpenML\\'
dataDir = 'D:\\mblachnik\\datasets\\datasets_ppe_tree\\'
datasets = [

    # "Agrawal1",
    # "Stagger1", #100% dokładności
    # "BayesianNetworkGenerator_spambase",
    # "BNG_sonar",

    (dataDir,"codrnaNorm"),
    (dataDir,"electricity-normalized"),
    (dataDir,"covtype"),
    (dataDir, "phpvcoG8S"),
    (dataDir, "Agrawal1"),
    (dataDir, "shuttle2"),
    (dataDir, "banana"),
    (dataDir, "coil2000"),  # Mocno niezbalansowany
    (dataDir, "magic"),
    (dataDir, "phoneme"),
    (dataDir, "ring"),
    (dataDir, "spambase"),
    (dataDir, "twonorm"),
    (dataDir, "titanic"),

    # "shuttle2"
]

all_res = []
tts = []
for max_depth in [2,3,4,5,6,7,10]:


    base_estimator = DecisionTreeClassifier(max_depth=max_depth)
    base_explainable_estimator = RandomForestClassifier(max_depth=10,n_estimators=100,n_jobs=10)
    #base_estimator = DecisionTreeClassifier(max_depth=max_depth, max_features='sqrt')  # RandomForestClassifier()
    #base_estimator = Pipeline([("FeatureTransformer", LinearDiscriminantAnalysis(n_components=1)),("RF", base_estimator)])
    #base_estimator = Pipeline([("FeatureTransformer", PCA(n_components=2)),("RF", base_estimator)])
    ite = 15
    models = [
        # ("PE", ppe.PPE_Classifier(type="pe",
        #                                 base_estimator=base_estimator,
        #                                 proto_selection=SimpleClusterCentroids(n_clusters=20),
        #                                 unbalanced_rate=0.2,
        #                                 minimum_regions=2,
        #                                 min_support=100,
        #                                 n_jobs=6)),
            ("PPE2", ppe.PPE_Classifier(type="ppe2",
                                        base_estimator=clone(base_estimator),
                                        #proto_selection=ClusterCentroids(sampling_strategy={-1:5,1:5}),
                                        proto_selection=ClusterCentroids(sampling_strategy={0: 10, 1: 10}),
                                        unbalanced_rate=0.2,
                                        minimum_regions=2,
                                        min_support=400,
                                        n_jobs=6)),
        #         ("PPE3",ppe.PPE_Classifier(type="ppe3",
        #                                 base_estimator=clone(base_estimator),
        #                                 proto_selection=ClusterCentroids(sampling_strategy={0:10,1:10}),
        #                                 unbalanced_rate=0.2,
        #                                 minimum_regions=2,
        #                                 min_support=400,
        #                                 prune_regions = False,
        #                                 n_jobs=6)),
                # ("EPPE", ppe.EPPE_Classifier(ppe_estimator=
                #                         ppe.PPE_Classifier(type="ppe3",
                #                             base_estimator=base_estimator,
                #                             proto_selection=SimpleClusterCentroids(n_clusters=10),
                #                             unbalanced_rate=0.2,
                #                             minimum_regions=2,
                #                             min_support=400,
                #                             prune_regions = False,
                #                             n_jobs=6),
                #                          n_estimators=10)),
                ("Tree", clone(base_estimator)),
                #("RF", RandomForestClassifier(max_depth=max_depth))
                ]

    for dirName,fName in datasets:
        #df = pd.read_csv(dirName + fName + "\\" + fName + ".dat",sep=";")
        df = pd.read_csv(dirName + fName + ".csv", sep=";",quoting=1, quotechar='"')
        X = df[[column for column in df.columns if column not in ["LABEL","id"]]]
        y = np.squeeze(df[['LABEL']].values)
        ohe = LabelEncoder()
        y = ohe.fit_transform(y)
        ohe = OneHotEncoder(sparse_output=False)
        sym_cols = (X.dtypes == "O").values
        ct = ColumnTransformer([('ohe',ohe, sym_cols)],remainder = 'passthrough')
        X = ct.fit_transform(X)
        # Xs = X.loc[:, X.dtypes== "O"]
        # Xn = X.loc[:, X.dtypes != "O"].values
        # Xst = ohe.fit_transform(Xs)
        # if (Xn.shape[1]>0) and (Xst.shape[1]>0):
        #     X = np.hstack((Xn, Xst.todense()))
        # elif (Xst.shape[1]>0):
        #     X = Xst.todense()
        # elif (Xn.shape[1]>0):
        #     X = Xn
        explainable_estimator = clone(base_explainable_estimator)
        explainable_estimator.fit(X,y)
        y = explainable_estimator.predict(X)
        print("Fitting")
        for model_name, model in models:
            model.fit(X,y)
            print(f"  Fitting {model_name}")
            if type(model)==ppe.PPE_Classifier:
                print(model.region_stats)
            yp = model.predict(X)
            acc =  accuracy_score(y_true=y, y_pred=yp)  #
            bacc = balanced_accuracy_score(y_true=y, y_pred=yp)  #
            f1 = f1_score(y_true=y, y_pred=yp)  #

            print(f"{model_name}: {[acc, bacc, f1]}")
            res = {"model": model_name,
                   "dataset": fName,
                   "acc":acc,
                   "bacc":bacc,
                   "f1":f1,
                   "depth": max_depth,
                   }
            if type(model)==ppe.PPE_Classifier:
                res["regions"] = model.region_stats.shape[0]
            all_res.append(res)



df_res = pd.DataFrame(all_res)

df_res.to_excel(f"Data/Results/results_ppe2_explainable_rf-{ite}.xlsx")


    # key = list(models[1][1].fitted_base_models_.keys())[0]
    # plt.figure(1)
    # tree.plot_tree(models[1][1].fitted_base_models_[key])
    # plt.show()
