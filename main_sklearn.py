# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:08:25 2023

@author: Marcin
"""
#ToDo: https://github.com/omadson/fuzzy-c-means/blob/master/fcmeans/main.py
import pandas as pd
import numpy as np
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
cv = StratifiedKFold(random_state=1, n_splits=10, shuffle=True)

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

    (dataDir, "banana"),
    #(dataDir, "coil2000"),
    #(dataDir, "magic"),
    #(dataDir, "phoneme"),
    (dataDir, "ring"),
    (dataDir, "spambase"),
    #(dataDir, "twonorm"),

    # "shuttle2"
]

all_res = []

for max_depth in [2,3,4,5,6,10]:


    base_estimator = DecisionTreeClassifier(max_depth=max_depth) #RandomForestClassifier()
    #base_estimator = Pipeline([("FeatureTransformer", LinearDiscriminantAnalysis(n_components=1)),("RF", base_estimator)])
    #base_estimator = Pipeline([("FeatureTransformer", PCA(n_components=2)),("RF", base_estimator)])

    models = [("PE", ppe.PPE_Classifier(type="pe",
                                        base_estimator=base_estimator,
                                        proto_selection=SimpleClusterCentroids(n_clusters=20),
                                        min_support=400,
                                        unbalanced_rate=0.2,
                                        minimum_regions=2,
                                        n_jobs=6)),
            # ("PPE2", ppe.PPE_Classifier(type="ppe2",
            #                             base_estimator=base_estimator,
            #                             proto_selection=ClusterCentroids(sampling_strategy={-1:5,1:5}),
            #                             unbalanced_rate=0.2,
            #                             minimum_regions=2,
            #                             min_support=400,
            #                             n_jobs=6)),
                ("PPE3",ppe.PPE_Classifier(type="ppe3",
                                        base_estimator=base_estimator,
                                        proto_selection=ClusterCentroids(sampling_strategy={0:10,1:10}),
                                        unbalanced_rate=0.2,
                                        minimum_regions=2,
                                        min_support=400,
                                        prune_regions = False,
                                        n_jobs=6)),
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
                ("Tree", base_estimator),
                ("RF", RandomForestClassifier())
                ]

    for dirName,fName in datasets:
        df = pd.read_csv(dirName + fName + "\\" + fName + ".dat",sep=";")
        X = df[[column for column in df.columns if column not in ["LABEL","id"]]]
        y = np.squeeze(df[['LABEL']].values)
        ohe = LabelEncoder()
        y = ohe.fit_transform(y)

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

df_res = pd.DataFrame(all_res)
df_res.to_excel(f"Data/Results/results_tree.xlsx")

    # key = list(models[1][1].fitted_base_models_.keys())[0]
    # plt.figure(1)
    # tree.plot_tree(models[1][1].fitted_base_models_[key])
    # plt.show()
