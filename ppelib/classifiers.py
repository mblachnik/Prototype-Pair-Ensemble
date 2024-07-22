import numpy as np
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils.estimator_checks import check_estimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import resample, gen_batches, check_random_state, check_X_y
from sklearn.decomposition import PCA

from imblearn.base import SamplerMixin
from ppelib.ppe import PPE, PPE2, PPE3, PE
from scipy.spatial.distance import cdist

from joblib import Parallel, delayed
import copy





class PPE_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 base_estimator=RandomForestClassifier(),
                 type="ppe",
                 unbalanced_rate=0.3,
                 min_support=500,
                 minimum_regions=1,
                 proto_selection={0: 10, 1: 10},
                 prune_regions=True,
                 n_jobs=None):
        """
        Constructor for the PPE_ensemble class.
        The idea of this algorithm is presented in (to appear)
        Basically it splits the datasets into smaller once using a distance to the nearest reference prototypes from
        to separate classes
        :param base_estimator: the base estimator to use for fitting the within each regino
        :param unbalanced_rate: if a region has unbalanced number of samples from the two classes the the region will be
         merged with another region that exist. Here we calculate it as min(c1/c2,c2/c1) so it shows the ration of the minority to majority class within region
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
        self.type = type
        self.minimum_regions = minimum_regions = 2
        self.prune_regions = prune_regions
        self.n_jobs = n_jobs

    def _initialize_ppe(self, X, y):
        if type(self.proto_selection) == dict:
            idx_all = np.zeros((y.shape[0]), dtype=bool)
            for label, n_samples in self.proto_selection.items():
                idClass = np.nonzero(y == label)[0]
                idx = sklearn.utils.resample(np.arange(idClass.shape[0]), n_samples=n_samples, replace=False)
                idx_all[idClass[idx]] = True
            Xp = X[idx_all, :]  # X of selected prototypes
            yp = y[idx_all]  # Y of selected prototypes
        elif issubclass(type(self.proto_selection), SamplerMixin):
            Xp, yp = self.proto_selection.fit_resample(X, y)
        else:
            raise ValueError("Unknown prototype selection method")

        if self.type == "ppe":
            ppe = PPE(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                      minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions)
        elif self.type == "ppe2":
            ppe = PPE2(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                       minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions)
        elif self.type == "ppe3":
            ppe = PPE3(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                       minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions)
        elif self.type == "pe":
            ppe = PE(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support, prune_regions=True,
                     minimum_n_regions=self.minimum_regions)
        else:
            raise ValueError("Unknown PPE type. Only (ppe,ppe2,pe) are avaliable")

        return ppe

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray):
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

        ppe = self._initialize_ppe(X, y)
        self.proto_ensemble_ = ppe

        regions = ppe.generate_regions(X, y)
        # pairs = ppe.assign_regions(X,regions)
        # _ux_regions, ux_regions_counts = np.unique(pairs, return_counts=True)
        # assert np.all(np.sort(ux_regions) == np.sort(_ux_regions))
        self.regions_ = list(regions.keys())
        self.region_stats = ppe.region_stats
        self.fitted_base_models_ = {}
        modelsInputData = []
        for region in regions:
            id = regions[region]
            if np.sum(id) == 0: continue
            Xm = X[id, :]
            ym = y[id]
            model = copy.deepcopy(self.base_estimator)
            modelsInputData.append((region, Xm, ym, model))

        if self.n_jobs is not None:
            parrTrainFun = lambda region, Xm, ym, model: (region, model.fit(Xm, ym))
            with Parallel(n_jobs=self.n_jobs) as parallel:
                res_all = parallel(delayed(parrTrainFun)(*input) for input in modelsInputData)
                self.fitted_base_models_ = {region: model for region, model in res_all}
        else:
            self.fitted_base_models_ = {region: model.fit(Xm, ym) for region, Xm, ym, model in modelsInputData}
        return self

    def predict(self, X: pd.DataFrame | np.ndarray):
        """
        Method used for predicting the output of the model. For each sample in X it determines the nearest region out of
         the existing region. And then based on the index of existing region it takes the classifier and performs prediction
        :param X: samples to be classified
        :return: predicted labels
        """
        check_is_fitted(self)
        X = check_array(X)
        regions_ = self.regions_
        sample2region = self.proto_ensemble_.assign_regions(X, regions_)  # For each sample in X get its nearest region
        yp = np.zeros(X.shape[0], dtype=int)  # Allocate memory
        for region in regions_:  # Iterate over reginos
            id = sample2region[region]  # Get samples which belong to region pair
            Xm = X[id, :]
            if Xm.size:
                model = self.fitted_base_models_[region]  # Take the classifier associated to region "pair"
                yp[id] = model.predict(Xm)  # Make prediction using the classifier assigned to region "pair"
        return yp


class EPPE_Classifier(VotingClassifier):
    def __init__(self,
                 ppe_estimator: PPE_Classifier,
                 n_estimators: int = 10,
                 voting="hard",
                 weights=None,
                 n_jobs=None,
                 flatten_transform=True,
                 verbose=False,
                 ):
        self.ppe_estimator = ppe_estimator
        self.n_estimators = n_estimators
        estimators = [(f"PPE_{i}", clone(ppe_estimator)) for i in range(n_estimators)]
        super().__init__(estimators=estimators,
                         voting=voting,
                         weights=weights,
                         n_jobs=n_jobs,
                         flatten_transform=flatten_transform,
                         verbose=verbose)


class RandomOracle(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, min_size):
        self.base_estimator = base_estimator
        self.min_size = min_size

    def _splitData(self, X):
        d = cdist(X, self.proto_, 'sqeuclidean')
        id = np.argmin(d, axis=1)
        return id == 0,id == 1

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        X, y = check_X_y(X, y)
        chk = True
        while chk:
            chk = True
            idx = np.random.randint(X.shape[0], size=2)
            proto = X[idx, :].copy()
            self.proto_ = proto
            id1,id2 = self._splitData(X)
            ux1,co_ux1 = np.unique(y[id1],return_counts=True)
            ux2, co_ux2 = np.unique(y[id2], return_counts=True)
            cl = len(self.classes_)
            c1 = len(ux1)
            c2 = len(ux2)
            if (cl==c1) and (c2==cl):
                chk = False
                for co1,co2 in zip(co_ux1,co_ux2):
                    if (co1<self.min_size) or (co2<self.min_size):
                        chk = True
                        continue


        X1 = X[id1,:]
        y1 = y[id1]
        X2 = X[id2,:]
        y2 = y[id2]
        self.model1_ = clone(self.base_estimator)
        self.model2_ = clone(self.base_estimator)
        self.model1_.fit(X1,y1)
        self.model2_.fit(X2,y2)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        id1,id2 = self._splitData(X)
        X1 = X[id1, :]
        X2 = X[id2, :]
        yp1 = self.model1_.predict(X1)
        yp2 = self.model2_.predict(X2)
        yp = np.zeros((X.shape[0],),dtype=int)
        yp[id1] = yp1
        yp[id2] = yp2
        return yp

class RandomOracle_Classifier(VotingClassifier):
    def __init__(self,
                 base_estimator: RandomOracle(base_estimator=RandomForestClassifier(),min_size=100),
                 n_estimators: int = 30,
                 voting="hard",
                 weights=None,
                 n_jobs=None,
                 flatten_transform=True,
                 verbose=False,
                 ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        estimators = [(f"RandomOracle_{i}", clone(base_estimator)) for i in range(n_estimators)]
        super().__init__(estimators,
                         voting=voting,
                         weights=weights,
                         n_jobs=n_jobs,
                         flatten_transform=flatten_transform,
                         verbose=verbose)

def random_feature_subsets(array, batch_size, random_state=1234):
    """ Generate K subsets of the features in X """
    random_state = check_random_state(random_state)
    features = list(range(array.shape[1]))
    random_state.shuffle([x for x in features])
    for batch in gen_batches(len(features), batch_size):
        yield features[batch]


class RotationTreeClassifier(DecisionTreeClassifier):
    def __init__(self,
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=1.0,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None,
    ):

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo

        super(RotationTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
        )

    def rotate(self, X):
        if not hasattr(self, 'rotation_matrix'):
            raise Exception('The estimator has not been fitted')

        return np.dot(X, self.rotation_matrix)

    def pca_algorithm(self):
        """ Deterimine PCA algorithm to use. """
        if self.rotation_algo == 'randomized':
            return PCA(svd_solver='randomized', random_state=self.random_state)
        elif self.rotation_algo == 'pca':
            return PCA()
        else:
            raise ValueError("`rotation_algo` must be either "
                             "'pca' or 'randomized'.")

    def _fit_rotation_matrix(self, X):
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        self.rotation_matrix = np.zeros((n_features, n_features),
                                        dtype=np.float32)
        for i, subset in enumerate(
                random_feature_subsets(X, self.n_features_per_subset,
                                       random_state=self.random_state)):
            # take a 75% bootstrap from the rows
            x_sample = resample(X, n_samples=int(n_samples * 0.75),
                                random_state=10 * i)
            pca = self.pca_algorithm()
            pca.fit(x_sample[:, subset])
            self.rotation_matrix[np.ix_(subset, subset)] = pca.components_

    def fit(self, X, y, sample_weight=None, check_input=True):
        self._fit_rotation_matrix(X)
        super(RotationTreeClassifier, self).fit(self.rotate(X), y,
                                                sample_weight, check_input)

    def predict_proba(self, X, check_input=True):
        return super(RotationTreeClassifier, self).predict_proba(self.rotate(X),
                                                                 check_input)

    def predict(self, X, check_input=True):
        return super(RotationTreeClassifier, self).predict(self.rotate(X),
                                                           check_input)

    def apply(self, X, check_input=True):
        return super(RotationTreeClassifier, self).apply(self.rotate(X),
                                                         check_input)

    def decision_path(self, X, check_input=True):
        return super(RotationTreeClassifier, self).decision_path(self.rotate(X),
                                                                 check_input)


class RotationForestClassifier(ForestClassifier):
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=1.0,
                 max_leaf_nodes=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RotationForestClassifier, self).__init__(
            estimator=RotationTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("n_features_per_subset", "rotation_algo",
                              "criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


class RotationTreeRegressor(DecisionTreeRegressor):
    def __init__(self,
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=1.0,
                 random_state=None,
                 max_leaf_nodes=None):

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo

        super(RotationTreeRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state)

    def rotate(self, X):
        if not hasattr(self, 'rotation_matrix'):
            raise Exception('The estimator has not been fitted')

        return np.dot(X, self.rotation_matrix)

    def pca_algorithm(self):
        """ Deterimine PCA algorithm to use. """
        if self.rotation_algo == 'randomized':
            return PCA(svd_solver='randomized', random_state=self.random_state)
        elif self.rotation_algo == 'pca':
            return PCA()
        else:
            raise ValueError("`rotation_algo` must be either "
                             "'pca' or 'randomized'.")

    def _fit_rotation_matrix(self, X):
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        self.rotation_matrix = np.zeros((n_features, n_features),
                                        dtype=np.float32)
        for i, subset in enumerate(
                random_feature_subsets(X, self.n_features_per_subset,
                                       random_state=self.random_state)):
            # take a 75% bootstrap from the rows
            x_sample = resample(X, n_samples=int(n_samples * 0.75),
                                random_state=10 * i)
            pca = self.pca_algorithm()
            pca.fit(x_sample[:, subset])
            self.rotation_matrix[np.ix_(subset, subset)] = pca.components_

    def fit(self, X, y, sample_weight=None, check_input=True):
        self._fit_rotation_matrix(X)
        super(RotationTreeRegressor, self).fit(self.rotate(X), y,
                                               sample_weight, check_input)

    def predict_proba(self, X, check_input=True):
        return super(RotationTreeRegressor, self).predict_proba(self.rotate(X),
                                                                check_input)

    def predict(self, X, check_input=True):
        return super(RotationTreeRegressor, self).predict(self.rotate(X),
                                                          check_input)

    def apply(self, X, check_input=True):
        return super(RotationTreeRegressor, self).apply(self.rotate(X),
                                                        check_input)

    def decision_path(self, X, check_input=True):
        return super(RotationTreeRegressor, self).decision_path(self.rotate(X),
                                                                check_input)


class RotationForestRegressor(ForestRegressor):
    def __init__(self,
                 n_estimators=10,
                 criterion="mse",
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=1.0,
                 max_leaf_nodes=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RotationForestRegressor, self).__init__(
            base_estimator=RotationTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("n_features_per_subset", "rotation_algo",
                              "criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes



