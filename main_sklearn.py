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

from ppelib import ppe




df = pd.read_csv("Data/banana.csv")
X = df[["X1", "X2"]]
y = np.squeeze(df[['Y']].values)
m1 = ppe.PPE_Classifier(proto_selection={-1:5, 1:5})
m2 = RandomForestClassifier()
#m3 = ppe.EPPE_Classifier(ppe_estimator=ppe.PPE_Classifier(base_estimator=RandomForestClassifier(n_estimators=10),
m3 = ppe.EPPE_Classifier(ppe_estimator=ppe.PPE_Classifier(base_estimator=DecisionTreeClassifier(),
                         proto_selection={0:5, 1:5}), #Warning: Here it must be class 0,1 instead of -1,1 becouse VotingEnsemble use onehot label encodings which converts output labels into values [0,1]
                         n_estimators=10)
m1.fit(X, y)
yp = m1.predict(X)
m3.fit(X, y)
yp3 = m3.predict(X)
print(f"Accuracy of PPE on the training set: {np.sum(y==yp)/yp.shape[0]}")
print(f"Accuracy of EPPE on the training set: {np.sum(y==yp3)/yp3.shape[0]}")
scores1 = cross_val_score(m1, X, y, cv=5)
scores2 = cross_val_score(m2, X, y, cv=5)
scores3 = cross_val_score(m3, X, y, cv=5)
print(f"PPE: {[np.mean(scores1)*100, np.std(scores1)*100]}")
print(f"RF:  {[np.mean(scores2)*100, np.std(scores2)*100]}")
print(f"EPPE:  {[np.mean(scores3)*100, np.std(scores3)*100]}")

