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

def treeAnalyzer(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    # for i in range(n_nodes):
    #     if is_leaves[i]:
    #         print(
    #             "{space}node={node} is a leaf node with value={value}.".format(
    #                 space=node_depth[i] * "\t", node=i, value=values[i]
    #             )
    #         )
    #     else:
    #         print(
    #             "{space}node={node} is a split node with value={value}: "
    #             "go to node {left} if X[:, {feature}] <= {threshold} "
    #             "else to node {right}.".format(
    #                 space=node_depth[i] * "\t",
    #                 node=i,
    #                 left=children_left[i],
    #                 feature=feature[i],
    #                 threshold=threshold[i],
    #                 right=children_right[i],
    #                 value=values[i],
    #             )
    #         )
    n_leaves = np.sum(is_leaves)
    return {"n_leaves":n_leaves,
            "n_decision_nodes":n_nodes - n_leaves}


#dataDir = 'D:\\Projects\\DataMining\\Data\\CSV_Experiments\\Above_1k\\'
#dataDirLarge = 'D:\\Projects\\DataMining\\Data\\Large - OpenML\\'
#dataDir = 'D:\\mblachnik\\datasets\\datasets_ppe_tree\\'
dataDir = 'Y:\\Datasets\\datasets_ppe\\'
datasets = [

    # "Agrawal1",
    # "Stagger1", #100% dokładności
    # "BayesianNetworkGenerator_spambase",
    # "BNG_sonar",

    # (dataDir,"codrnaNorm"),
    # (dataDir,"electricity-normalized"),
    # (dataDir,"covtype"),
    # (dataDir, "phpvcoG8S"),
    # (dataDir, "Agrawal1"),
    # (dataDir, "shuttle2"),
    (dataDir, "banana"),
    # (dataDir, "coil2000"),  # Mocno niezbalansowany
    # (dataDir, "magic"),
    # (dataDir, "phoneme"),
    # (dataDir, "ring"),
    # (dataDir, "spambase"),
    # (dataDir, "twonorm"),
    # (dataDir, "titanic"),

    # "shuttle2"
]

all_res = []
tts = []
do_ttest = False
do_cv = True
for min_support in [100,300,700]:
    for n_clusters in [3,5,7,10,15]:
        max_depth = 3
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0)
        base_estimator = rf
        #base_estimator = DecisionTreeClassifier(max_depth=max_depth)
        #base_estimator = DecisionTreeClassifier(max_depth=max_depth, max_features='sqrt')  # RandomForestClassifier()
        #base_estimator = Pipeline([("FeatureTransformer", LinearDiscriminantAnalysis(n_components=1)),("RF", base_estimator)])
        #base_estimator = Pipeline([("FeatureTransformer", PCA(n_components=2)),("RF", base_estimator)])
        ite = 100
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
                                            proto_selection=ClusterCentroids(sampling_strategy={0: n_clusters, 1: n_clusters}),
                                            unbalanced_rate=0.2,
                                            minimum_regions=2,
                                            min_support=min_support,
                                            n_jobs=3)),
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
                    #("Tree", clone(base_estimator)),
                    ("RF", clone(rf))
                    #("RF2", RandomForestClassifier(n_jobs=10))
                    ]

        for dirName,fName in datasets:
            print(f"Processing: {fName} with depth: {max_depth}")
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

            # print("Fitting")
            # for model_name, model in models:
            #     model.fit(X,y)
            #     print(f"  Fitting {model_name}")
            #     if type(model)==ppe.PPE_Classifier:
            #         print(model.region_stats)
            # print("Predicting")
            # for model_name, model in models:
            #     yp = model.predict(X)
            #     print(f"  Accuracy of {model_name} on the training set: {np.mean(y == yp)}")

            print("CV")
            scores=[]
            for model_name, model in models:
                res = {"model": model_name,
                       "dataset": fName}
                if do_cv:
                    cv = StratifiedKFold(random_state=1, n_splits=10, shuffle=True)
                    score = cross_val_score(clone(model), X, y, cv=cv, n_jobs=10)
                    me = np.mean(score) * 100
                    st = np.std(score) * 100
                    print(f"{model_name}: {[me, st]}")
                    res_tmp = {"me":me,
                           "std":st,
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
                           }
                    res.update(res_tmp)
                    scores.append((model_name, score))
                tmp_model = clone(model)
                if type(model)==ppe.PPE_Classifier:
                    tmp_model : ppe.PPE_Classifier
                    tmp_model.fit(X, y)
                    res["regions"] = tmp_model.region_stats.shape[0]
                    res["n_clusters"] = n_clusters
                    res["min_support"] = min_support
                    #trees = tmp_model.fitted_base_models_.values()
                    #trees_cmomlexity = [treeAnalyzer(tree) for tree in trees]
                    #tree_eq = {"n_leaves":0,"n_decision_nodes":0}
                    #for cpl in trees_cmomlexity:
                    #    tree_eq["n_leaves"]+=cpl["n_leaves"]
                    #    tree_eq["n_decision_nodes"]+=cpl["n_decision_nodes"]
                    #res.update(tree_eq)

                if type(model)==DecisionTreeClassifier:
                    tmp_model.fit(X, y)
                    res.update(
                        treeAnalyzer(tmp_model)
                    )

                all_res.append(res)

            if do_ttest:
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
df_res.to_excel(f"Data/Results/results_ppe3_tree-{ite}.xlsx")

if do_ttest:
    df_tt = pd.DataFrame(tts)
    df_res.to_excel(f"Data/Results/results_ppe3_tree_tt-{ite}.xlsx")

    # key = list(models[1][1].fitted_base_models_.keys())[0]
    # plt.figure(1)
    # tree.plot_tree(models[1][1].fitted_base_models_[key])
    # plt.show()
