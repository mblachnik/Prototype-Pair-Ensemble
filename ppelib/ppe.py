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
        """
        Unpair of the Cantar function.
        Normally for given value it calculates two integer coordinates.
        Here z can be a single value or an array of values, where each
        to each value of z the unpair function will be applied
        :param z: the values to be unpaired
        :return: a tuple of two values or two arrays of values
        """
        t = int(math.floor((math.sqrt(8 * z + 1) - 1) / 2))
        x = int(t * (t + 3) / 2 - z)
        y = int(z - t * (t + 1) / 2)
        return x, y

    @staticmethod
    def pairCantor(a, b):
        """
        Pairing function using Cantar formula
        For a given pair of integer values it combines them into a single int value
        Here a and b can be arrays of two integer where each pair of values a[0] b[0] will be paired using Cantar formula
        :param a: value or array of values to be one half of pair
        :param b: value or array of values to be one half of pair
        :return: bired values
        """
        return 0.5 * (a + b) * (a + b + 1) + b

    def assign_regions(self, X: np.array, pairs: np.array) -> np.array:
        """
        For given samples in X it assignes new samples to one of hte regions
        :param X: input data where each row will be assigned to one of existing pairs
        :param pairs: a list of unique pairs
        :return: the nearest pair for each row in X
        """
        d = cdist(X, self.proto, metric="sqeuclidean")
        ds = np.zeros((d.shape[0], len(pairs))) #Allocate memory to store the results - distances to prototypes constituting given pair
        i = 0
        for p in pairs:
            a, b = self.unpairCantor(p) #Get indexes of prototypes of a pair
            ds[:, i] = d[:, a] + d[:, b] #Get the distance to the pair, note that here i denotes the index of a given pair
            i += 1
        idp = np.argmin(ds, axis=1) #Find smallest distances ang get index of this nearest pairs
        out = pairs[idp] #Convert a list of unique pairs to the full array of new pairs
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
        if isinstance(X, pd.DataFrame): #If dataframe then convert input data to numpy
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values #If pd.DataFrame then convert input data to numpy
        self.ux = ux #Get labels
        if len(ux) != 2: #If more then 2 labels then error - the algorithm only supports 2 class problems
            raise ValueError("The algorithm assums binary classification, but the number of prototype classes is != 2")
        PY = self.proto_labels
        P = self.proto
        # indexes of samples from given class
        idPos = np.squeeze(PY == ux[0]) #Samples from first class
        idNeg = np.squeeze(PY == ux[1]) #Samples from second class
        # for each sample in X it gets nearest samples from both classes
        dPos = cdist(X, P[idPos, :], metric="sqeuclidean") #Calculate distance from X to the prototypes from positive class
        dNeg = cdist(X, P[idNeg, :], metric="sqeuclidean") #Calculate distance from X to the prototypes from negative class
        npp = np.argmin(dPos, axis=1)  # Get index of the Nearest prototype positive
        npn = np.argmin(dNeg, axis=1)  # Get index of the Nearest prototype negative

        idPosI = np.nonzero(np.squeeze(idPos))[0] #Convert binary index into numeric one for positive samples
        idNegI = np.nonzero(np.squeeze(idNeg))[0] #Convert binary index into numeric one for negative samples

        idPosN = idPosI[npp]
        idNegN = idNegI[npn]

        # Change order so that smaller number is always first to avoid duplicated indees such that from one sample the
        # neares pair is 2, 10, and for the other 10, 2. To avoid it we always start with the smalles index so in both cases that will be 2, 10
        id = idPosN > idNegN
        tmp = idPosN[id]
        idPosN[id] = idNegN[id]
        idNegN[id] = tmp

        # Pair
        pairs = self.pairCantor(idPosN, idNegN)
        stats = self.__getRegionStats(X, y, pairs)
        ux_pairs = np.unique(pairs)
        # If samples do not fulfill given statisitcs, reasign these samples to one of existing regions
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
        """
        Constructor for the PPE_ensemble class.
        The idea of this algorithm is presented in (to appear)
        Basically it splits the datasets into smaller once using a distance to the nearest reference prototypes from
        to separate classes
        :param base_estimator: the base estimator to use for fitting the within each regino
        :param unbalanced_rate: if a region has unbalanced number of samples from the two classes the the region will be
         merged with another region that exist. Here we define the unbalanced tation
        :param min_support: minimum number of samples per region
        :param proto_selection: a protothpe selectin method. Possible options are:
            dict wher kays are class label names and values represent the number of prototypes which will be randomly
            selected from samples of given class
            an algorithm from imblearn packated or some other algorithm which supports fit_resample method
        """
        self.base_estimator = base_estimator
        self.unbalanced_rate = unbalanced_rate
        self.min_support = min_support
        self.proto_selection = proto_selection

    def fit(self, X, #:pd.DataFrame | np.array,
                 y):#:pd.DataFrame | np.array):
        """
        Train and algorithm, it first starts be identifing region and then for each region it trains the base model
        The trained models are stored in self.fitted_base_models_ attribute which is a dict where keys are prototype
        pairs (see Cantar pairing function, or PPE class) and values are traind models
        :param X: training samples delivered as numpy arrays or a DataFrame
        :param y: training sample labels delivered as numpy arrays or a DataFrame
        :return: self - trained model
        """
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
        """
        Method used for predicting the output of the model. For each sample in X it determines the nearest region out of
         the existing region. And then based on the index of existing region it takes the classifier and performs prediction
        :param X: samples to be classified
        :return: predicted labels
        """
        check_is_fitted(self)
        X = check_array(X)
        ux_pairs = self.ux_protoPairs_
        pairs = self.ppe_.assign_regions(X,ux_pairs) #For each sample in X get its nearest region
        yp = np.zeros(X.shape[0]) #Allocate memory
        for pair in ux_pairs: #Iterate over reginos
            id = pairs==pair #Get samples which belong to region pair
            Xm = X[id, :]
            model = self.fitted_base_models_[pair] #Take the classifier associated to region "pair"
            yp[id] = model.predict(Xm) #Make prediction using the classifier assigned to region "pair"
        return yp