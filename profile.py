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
from ppelib.sampler.kmeans_sampler import SimpleClusterCentroids
from imblearn.under_sampling import ClusterCentroids

from ppelib import ppe




df = pd.read_csv("Data/banana.csv")
X = df[["X1", "X2"]]
y = np.squeeze(df[['Y']].values)

# m = ppe.PPE_Classifier(type="pe",
#                         proto_selection=SimpleClusterCentroids(n_clusters=20),
#                         min_support=400,
#                         unbalanced_rate=0.2,
#                         minimum_regions=2)
m = ppe.PPE_Classifier(type="ppe2",
                        proto_selection=ClusterCentroids(sampling_strategy={-1:15,1:15}),
                        unbalanced_rate=0.2,
                        minimum_regions=2,
                        min_support=400)
# m = ppe.EPPE_Classifier(ppe_estimator=
#                          ppe.PPE_Classifier(base_estimator=RandomForestClassifier(n_estimators=10),
#                                             proto_selection={0:5, 1:5}), #Warning: Here it must be class 0,1 instead of -1,1 becouse VotingEnsemble use onehot label encodings which converts output labels into values [0,1]
#                          n_estimators=10)
m.fit(X, y)
yp1 = m.predict(X)
print(f"Accuracy of PE (naive PPE without pairing) on the training set: {np.sum(y == yp1) / yp1.shape[0]}")


