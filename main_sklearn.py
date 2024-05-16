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


cv = StratifiedKFold(random_state=1, n_splits=10, shuffle=True)

dataDir = 'D:\\Projects\\DataMining\\Data\\CSV_Experiments\\Above_1k\\'
dataDirLarge = 'D:\\Projects\\DataMining\\Data\\Large - OpenML\\'
datasets = [

    # "Agrawal1",
    # "Stagger1", #100% dokładności
    # "BayesianNetworkGenerator_spambase",
    # "BNG_sonar",

    (dataDirLarge, "Agrawal1"),
    (dataDirLarge,"codrnaNorm"),
    (dataDirLarge,"electricity-normalized"),
    (dataDirLarge,"covtype"),
    (dataDirLarge,"shuttle2"),
    (dataDirLarge,"phpvcoG8S"),

    # (dataDir, "banana"),
    # (dataDir, "coil2000"),
    # #(dataDir, "magic"),
    # (dataDir, "phoneme"),
    # (dataDir, "ring"),
    # (dataDir, "spambase"),
    # (dataDir, "twonorm"),

    # "shuttle2"
]

all_res = []
tts = []
for max_depth in [3,4,5,6,7,10]:


    base_estimator = DecisionTreeClassifier(max_depth=max_depth) #RandomForestClassifier()
    #base_estimator = DecisionTreeClassifier(max_depth=max_depth, max_features='sqrt')  # RandomForestClassifier()
    #base_estimator = Pipeline([("FeatureTransformer", LinearDiscriminantAnalysis(n_components=1)),("RF", base_estimator)])
    #base_estimator = Pipeline([("FeatureTransformer", PCA(n_components=2)),("RF", base_estimator)])
    ite = 1
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
            #     ("PPE3",ppe.PPE_Classifier(type="ppe3",
            #                             base_estimator=base_estimator,
            #                             proto_selection=ClusterCentroids(sampling_strategy={0:10,1:10}),
            #                             unbalanced_rate=0.2,
            #                             minimum_regions=2,
            #                             min_support=400,
            #                             prune_regions = False,
            #                             n_jobs=6)),
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
                ("RF", RandomForestClassifier(max_depth=max_depth))
                ]

    for dirName,fName in datasets:
        #df = pd.read_csv(dirName + fName + "\\" + fName + ".dat",sep=";")
        df = pd.read_csv(dirName + fName + ".csv", sep=";",quoting=1, quotechar='"')
        X = df[[column for column in df.columns if column not in ["LABEL","id"]]]
        y = np.squeeze(df[['LABEL']].values)
        ohe = LabelEncoder()
        y = ohe.fit_transform(y)
        ohe = OneHotEncoder()
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

        print("Fitting")
        for model_name, model in models:
            model.fit(X,y)
            print(f"  Fitting {model_name}")
            if type(model)==ppe.PPE_Classifier:
                print(model.region_stats)
        #
        # print("Predicting")
        # for model_name, model in models:
        #     yp = model.predict(X)
        #     print(f"  Accuracy of {model_name} on the training set: {np.mean(y == yp)}")

        print("CV")
        #scores = {}
        scores=[]
        for model_name, model in models:
            score = cross_val_score(clone(model), X, y, cv=cv)
            #scores[model_name] = score
            me = np.mean(score) * 100
            st = np.std(score) * 100
            print(f"{model_name}: {[me, st]}")
            res = {"model": model_name,
                   "dataset": fName,
                   "me":me,
                   "std":st,
                   "depth": max_depth
                   }
            if type(model)==ppe.PPE_Classifier:
                res["regions"] = model.region_stats.shape[0]
            all_res.append(res)
            scores.append((model_name,score))
        model_name1,d1 = scores[0]
        pvs = []
        for i in range(1, len(scores)):
            model_name2, d2 = scores[i]
            tt = stats.ttest_rel(d1, d2)
            pvs.append(
                {"refName":model_name1,
                 "modName":model_name2,
                 "p-value":tt.pvalue,
                 "reject":tt.pvalue < 0.05,
                 "max_depth":max_depth}
            )
        tts.append(pvs)


df_res = pd.DataFrame(all_res)
df_tt = pd.DataFrame(tts)
df_res.to_excel(f"Data/Results/results_ppe_tree-{ite}.xlsx")
df_res.to_excel(f"Data/Results/results_ppe_tree_tt-{ite}.xlsx")

    # key = list(models[1][1].fitted_base_models_.keys())[0]
    # plt.figure(1)
    # tree.plot_tree(models[1][1].fitted_base_models_[key])
    # plt.show()
