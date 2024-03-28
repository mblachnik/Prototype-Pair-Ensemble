# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:08:25 2023

@author: Marcin
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.under_sampling import ClusterCentroids

from ppelib import ppe as ppelib
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

mpl.use("QtAgg")

def plotData(x, y, label1, label2=None, colors='rgb', markers=['.', '.'], markersize=5):
    print(df1.a1.shape)
    uxs = np.unique(label1)
    lux = len(uxs)
    lux2 = 1

    if label2 is not None:
        uxs2 = np.unique(label2)
        lux2 = len(uxs2)

    if len(markers) == 1:
        markers = markers * (lux * lux2)

    for i, ux in enumerate(uxs):
        id1 = label1 == ux
        if label2 is not None:
            for j, ux2 in enumerate(uxs2):
                id2 = label2 == ux2
                id12 = id1 & id2
                print(np.sum(id12))
                plt.plot(x[id12], y[id12], color=colors[j], marker=markers[i], linestyle="None", markersize=markersize)
        else:
            print(np.sum(id1))
            plt.plot(x[id1], y[id1], color=colors[i], marker=markers[i], linestyle="None", markersize=markersize)




fName = "poly"
df1 = pd.read_csv('Data/Results/train_regions.csv', sep=";")
# df2 = pd.read_csv('Data/Results/proto_regions.csv',sep=";")

df11 = df1.copy()
# df1 = df1.sample(500,axis=0)

width, height = 8, 6

# ux_protoPairs = np.unique(df1['ID_Proto_Pair'])

X = df11[["a1", "a2"]]
y = df11[['Class']]
id1 = y == 1
n = 2
idx1 = id1[id1.values].sample(n=3)
idx2 = id1[~id1.values].sample(n=2)
PX = pd.concat((df11.loc[idx1.index, ["a1", "a2"]],
                df11.loc[idx2.index, ["a1", "a2"]]))  # df2[["a1","a2"]]
PY = pd.concat((df11.loc[idx1.index, ["Class"]],
                df11.loc[idx2.index, ["Class"]]))  # df2[["a1","a2"]]

mi = np.min(X, axis=0)
mx = np.max(X, axis=0)
limx = (mi.a1, mx.a1)
limy = (mi.a2, mx.a2)

# base_estimator = RandomForestClassifier()
# ppe = ppelib.PPE_Classifier(base_estimator=base_estimator,
#                                     type="ppe2",
#                                     proto_selection=ClusterCentroids(estimator=KMeans(random_state=0, n_init=10),
#                                         sampling_strategy={0: 15, 1: 15}),
#                                     min_support=400,
#                                     minimum_regions=2)

cl = ClusterCentroids(estimator=KMeans(random_state=0, n_init=10),
                  sampling_strategy={0: 3, 1: 3})
y[y==-1] = 0
#PX,PY =cl.fit_resample(X,y)
PX =pd.DataFrame([[0.99470783, 0.51770136],
       [0.02465749, 0.14812839],
       [0.61552881, 0.10683872],
       [0.16161304, 0.94875707],
       [0.75546858, 0.68521314]], columns=["a1","a2"]).reset_index(drop=True)

PY = pd.DataFrame([[ 1.],
       [ 1.],
       [ 1.],
       [0.],
       [0.]], columns=["Class"]).reset_index(drop=True)

ppe = ppelib.PPE2(proto=PX,
                  proto_labels=PY,
                  unbalanced_rate=0.05,
                  min_support=10,
                  prune_regions=True,
                  minimum_n_regions=1)
#    ppelib.PPE(P, PY,unbalanced_rate=0, min_support=1))
regions = ppe.generate_regions(X, y)

ux_protoPairs = list(regions.keys())
stats = ppe.region_stats

print(f"UX Proto F0:")
print(ux_protoPairs)
print("Counts")
print(stats)
print("==========")

q = ppe.assign_regions(X, ux_protoPairs)

xlist = np.linspace(limx[0], limx[1], 100)
ylist = np.linspace(limy[0], limy[1], 100)
Xc, Yc = np.meshgrid(xlist, ylist)

yc = np.reshape(Yc, (-1, 1))
xc = np.reshape(Xc, (-1, 1))
xyc = np.concatenate((xc, yc), axis=1)

qc = ppe.assign_regions(xyc, ux_protoPairs)
qcc = np.zeros((xyc.shape[0],1))
for i,(k,v) in enumerate(qc.items()):
    qcc[v]=i

qcc = np.reshape(qcc, Xc.shape)

n = len(ux_protoPairs)
n += 2  # Reserwujemy dodatkowe dwa kolory na klasy

colors = mpl.colormaps[
    # 'tab20'
    "gist_ncar"
].resampled(n)

cols = colors(range(n + 1))

cols2 = cols[[0, n]]
cols = cols[1:n]

# colors2 = cols[0:n-1]
# colors = cols[2:n]


plt.figure(1, figsize=(width, height))
plt.clf()
plotData(df1.a1, df1.a2, label1=df1["Class"],
         markers=['o'], colors=cols2, markersize=4)
plotData(PX.a1, PX.a2, PY.Class, markers=['*', 'o'], colors='rr', markersize=15)
PX.reset_index(inplace=True, drop=True)
for pair in ux_protoPairs:
    i, j = ppe.unpairCantor(pair)
    x = PX.loc[[i, j], "a1"]
    y = PX.loc[[i, j], "a2"]
    plt.plot(x, y, 'r')
ax = plt.gca()
p = PX  # = df2[["a1","a2"]].values
# p = np.vstack([p, [[0, 1],[1, 0]]])
vor = Voronoi(p)
voronoi_plot_2d(vor,
                ax=ax,
                show_points=False,
                show_vertices=False)


cp = plt.contourf(Xc, Yc, qcc, alpha=0.7, cmap="gist_ncar")  # colors=cols)
# plt.colormap(hot)
plt.xlim(limx)
plt.ylim(limy)
plt.show()
# plt.savefig(f'pic/regions_{fName}.png', bbox_inches='tight')


# plotData(df1.a1, df1.a2, , markers=['o','o'], colors='rr')
# plt.figure(2,figsize=(width,height))
# plt.clf()
# plotData(df11.a1, df11.a2, df11.Class, label2=df11.ID_Proto_Pair, markers=['o','o'],colors=colors,markersize=10)
# plotData(df2.a1, df2.a2, df2.Class, markers=['*','o'], colors='rr',markersize=10)
# plt.savefig(f"pic/data_{fName}.png", bbox_inches='tight')
# ax = plt.axis()
# plt.figure(1)
# plt.axis(ax)
# plt.show()


# for pair in protoPairs:
#     i,j = unpair(pair)
#     x = df2.loc[[i,j],"a1"]
#     y = df2.loc[[i,j],"a2"]
#     plt.plot(x,y,'b')
# ax = plt.gca()
# p = df2[["a1","a2"]].values
# p = np.vstack([p, [[0, 1],[1, 0]]])
# vor = Voronoi(p)
# voronoi_plot_2d(vor,
#                 ax=ax,
#                 show_points=False,
#                 show_vertices=False)
# plotData(df1.a1, df1.a2, , markers=['o','o'], colors='rr')