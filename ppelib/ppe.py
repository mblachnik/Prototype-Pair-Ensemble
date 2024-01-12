import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math

import sklearn
from imblearn.base import SamplerMixin
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class PPE:
    def __init__(self, proto, proto_labels, unbalanced_rate=0.01, min_support=10):
        self.ux = None
        if isinstance(proto, pd.DataFrame):
            proto = proto.values
        if isinstance(proto, pd.DataFrame) or isinstance(proto_labels, pd.DataFrame):
            proto_labels = proto_labels.values
        self.proto = proto
        self.proto_labels = proto_labels
        self.unbalanced_rate = unbalanced_rate
        self.min_support = min_support

    @staticmethod
    def unpairCantor(z):
        t = int(math.floor((math.sqrt(8 * z + 1) - 1) / 2))
        x = int(t * (t + 3) / 2 - z)
        y = int(z - t * (t + 1) / 2)
        return x, y

    @staticmethod
    def pairCantor(a, b):
        return 0.5 * (a + b) * (a + b + 1) + b

    def assign_regions(self, X: np.array, pairs: np.array) -> np.array:
        """
        For given samples in X it assignes new samples to one of hte regions
        :param X:
        :param pairs:
        :return:
        """
        d = cdist(X, self.proto, metric="sqeuclidean")
        ds = np.zeros((d.shape[0], len(pairs)))
        i = 0
        for p in pairs:
            a, b = self.unpairCantor(p)
            ds[:, i] = d[:, a] + d[:, b]
            i += 1
        idp = np.argmin(ds, axis=1)
        out = pairs[idp]
        return out

    def __getRegionStats(self, X, y, pairs):
        """
        Function gets as inpyt labeled data, and region assigment and returns and calculates statists over the regions.
        This statistis in a form of a pandas Dataframe is returned where one column is a region_id (can be set as index),
         and the remining columns are: numer of samples in each class in each region (Class1, Class2) and rank which determines the quality
         of a region. The following rank values can be assigned:
         0 - a region contains samples of only one class
         1 - if a region has less samples the defined by the min_support = the minimum number of samples in a region
         2 - if relation between samples of one class over samples from another class is not above a certain threshold:
            np.minimum(s1/s2,s2/s1) < inbalanced_rate - its aim is to keep the number of samples at certain level
        represent statistics including
        :param X: training set
        :param y: labels of the training set
        :param pairs: regions id
        :return: a DataFrame with statistics for each region
        """
        inbalanced_rate = self.unbalanced_rate
        min_support = self.min_support
        # X.loc[:,"#PAIR#"] = pair
        # X.loc[:,"#PAIR#"] = pair
        ux_pairs = np.unique(pairs)
        ux_labels = self.ux

        stats = {"Pair": [],
                 'Class1': [],
                 'Class2': [],
                 'rank': []}
        for pair in ux_pairs:
            id = pairs == pair
            s1 = np.sum(y[id] == ux_labels[0])
            s2 = np.sum(y[id] == ux_labels[1])
            rank = 100
            if (s1 == 0) or (s2 == 0):
                rank = 0
            else:
                if np.sum(id) < min_support:
                    rank = 1
                else:
                    if np.minimum(s1 / s2, s2 / s1) < inbalanced_rate:
                        rank = 2
            stats['Pair'].append(pair)
            stats['Class1'].append(s1)
            stats['Class2'].append(s2)
            stats['rank'].append(rank)
        stats_df = pd.DataFrame(stats)
        return stats_df

    def __get_most_corrupted_regin(self, stats: pd.DataFrame()):
        stats.sort_values("rank", inplace=True)
        stats.reset_index(drop=True, inplace=True)
        if stats.loc[0, "rank"] < 100:
            return stats.loc[0, "Pair"]
        else:
            return -1

    def generate_regions(self,
                         X, #: pd.DataFrame | np.array,
                         y #: pd.DataFrame | np.array
                         ) -> np.array:
        """
        For input data and already knwon prototype pairs it assigns samples to given region
        :param X:
        :param y:
        :return:
        """

        ux = np.unique(self.proto_labels)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        self.ux = ux
        if len(ux) != 2:
            raise ValueError("The algorithm assums binary classification, but the number of prototype classes is != 2")
        PY = self.proto_labels
        P = self.proto
        # indexes of samples from given class
        idPos = np.squeeze(PY == ux[0])
        idNeg = np.squeeze(PY == ux[1])
        # for each sample in X it gets nearest samples from both classes
        dPos = cdist(X, P[idPos, :], metric="sqeuclidean")
        dNeg = cdist(X, P[idNeg, :], metric="sqeuclidean")
        npp = np.argmin(dPos, axis=1)  # Nearest prototype positive
        npn = np.argmin(dNeg, axis=1)  # Nearest prototype negative

        idPosI = np.nonzero(np.squeeze(idPos))[0]
        idNegI = np.nonzero(np.squeeze(idNeg))[0]

        idPosN = idPosI[npp]
        idNegN = idNegI[npn]

        # Change order so that smaller number is always first
        id = idPosN > idNegN
        tmp = idPosN[id]
        idPosN[id] = idNegN[id]
        idNegN[id] = tmp

        # Pair
        pairs = self.pairCantor(idPosN, idNegN)
        stats = self.__getRegionStats(X, y, pairs)
        ux_pairs = np.unique(pairs)
        # If samples fo not fulfill given statisitcs, reg
        while (pair := self.__get_most_corrupted_regin(stats)) != -1:
            ux_pairs = ux_pairs[ux_pairs != pair]
            pairs = self.assign_regions(X, ux_pairs)
            stats = self.__getRegionStats(X, y, pairs)
        return pairs


import copy


class PPE_ensemble(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 base_estimator=RandomForestClassifier(),
                 unbalanced_rate=0.01,
                 min_support=100,
                 proto_selection={0:10, 1:10}):
        self.base_estimator = base_estimator
        self.unbalanced_rate = unbalanced_rate
        self.min_support = min_support
        self.proto_selection = proto_selection

    def fit(self, X, #:pd.DataFrame | np.array,
                 y):#:pd.DataFrame | np.array):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if type(self.proto_selection) == dict:
            idx_all = np.zeros((y.shape[0]),dtype=bool)
            for label, n_samples in self.proto_selection.items():
                idClass = np.nonzero(y == label)[0]
                idx = sklearn.utils.resample(np.arange(idClass.shape[0]),n_samples=n_samples, replace=False)
                idx_all[idClass[idx]] = True
            Xp = X[idx_all,:]
            yp = y[idx_all]
        elif type(self.proto_selection) == SamplerMixin :
            Xp, yp = self.proto_selection.fit_resample(X, y)
        ppe = PPE(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support)
        self.ppe_ = ppe
        pairs = ppe.generate_regions(X, y)
        ux_protoPairs, ux_counts = np.unique(pairs, return_counts=True)
        self.pairs_ = ux_protoPairs
        self.pairs_count_ = ux_counts

        self.ux_protoPairs_ = ux_protoPairs
        self.fitted_base_models_ = {}
        for pair in ux_protoPairs:
            id = pairs == pair
            Xm = X[id, :]
            ym = y[id]

            model = copy.deepcopy(self.base_estimator)
            model.fit(Xm, ym)
            self.fitted_base_models_[pair] = model
        return self

    def predict(self, X, #:pd.DataFrame | np.array
                ):
        check_is_fitted(self)
        X = check_array(X)
        ux_pairs = self.ux_protoPairs_
        pairs = self.ppe_.assign_regions(X,ux_pairs)
        yp = np.zeros(X.shape[0])
        for pair in ux_pairs:
            id = pairs==pair
            Xm = X[id, :]
            model = self.fitted_base_models_[pair]
            yp[id] = model.predict(Xm)
        return yp