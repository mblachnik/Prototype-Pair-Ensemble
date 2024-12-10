import numpy as np
import sklearn
from collections import OrderedDict
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing
from imblearn.under_sampling.base import BaseUnderSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import _random_state_docstring
from ppelib.ppe import PPE, PPE2, PPE3, PE
from imblearn.base import SamplerMixin
import copy
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import resample, gen_batches, check_random_state, check_X_y
from joblib import Parallel, delayed
import time

VOTING_KIND = ("auto", "hard", "soft")
@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class ppeResample(BaseUnderSampler):
    ppe_init_time:float = 0.0
    ppe_fit_time:float = 0.0
    ppe_region_time:float = 0.0
    def __init__(self,
                 base_estimator,
                 ppe_type="ppe",
                 unbalanced_rate=0.3,
                 min_support=500,
                 minimum_regions=1,
                 proto_selection={0: 10, 1: 10},
                 prune_regions=True,
                 n_jobs=None,
                 sampling_strategy="auto"):
        super().__init__(sampling_strategy=sampling_strategy)
        self.base_estimator = base_estimator
        self.unbalanced_rate = unbalanced_rate
        self.min_support = min_support
        self.proto_selection = proto_selection
        self.ppe_type = ppe_type
        self.minimum_regions = minimum_regions
        self.prune_regions = prune_regions
        self.n_jobs = n_jobs

    def _initialize_ppe(self, X, y):
        time_start = time.time()
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

        if self.ppe_type == "ppe":
            ppe = PPE(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                      minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions)
        elif self.ppe_type == "ppe2":
            ppe = PPE2(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                       minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions)
        elif self.ppe_type == "ppe3":
            ppe = PPE3(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                       minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions)
        elif self.ppe_type == "pe":
            ppe = PE(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support, prune_regions=True,
                     minimum_n_regions=self.minimum_regions)
        else:
            raise ValueError("Unknown PPE type. Only (ppe,ppe2,pe) are avaliable")
        self.ppe_init_time = time.time() - time_start
        return ppe
    
    
    # def __init__(self, max_runs:int, step_size:float, batch_size:int, prototype_n_per_class:int, sampling_strategy="auto"):
    #     super().__init__(sampling_strategy=sampling_strategy)
        
    
    def _fit_resample(self, X, y):
        X_selected = []
        y_selected = []
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        ppe = self._initialize_ppe(X, y)
        self.proto_ensemble_ = ppe
        time_start = time.time()
        regions = ppe.generate_regions(X, y)
        self.ppe_region_time = time.time() - time_start
        self.regions_ = list(regions.keys())
        self.region_stats = ppe.region_stats
        self.fitted_base_models_ = {}
        modelsInputData = []
        for region in regions:
            id = regions[region]
            if np.sum(id) == 0: continue
            Xm = X[id, :]
            ym = y[id]
            selector = copy.deepcopy(self.base_estimator)
            modelsInputData.append((region, Xm, ym, selector))
        
        if self.n_jobs is not None:
            parrTrainFun = lambda region, Xm, ym, selector: (region, selector.fit(Xm, ym))
            with Parallel(n_jobs=self.n_jobs, verbose=3) as parallel:
                res_all = parallel(delayed(parrTrainFun)(*input) for input in modelsInputData)
                self.fitted_base_models_ = {region: model for region, model in res_all}
        else:
            self.fitted_base_models_ = {region: model.fit(Xm, ym) for region, Xm, ym, model in modelsInputData}

        for selector in self.fitted_base_models_.values():
            # selector.fit(X,y)
            idx = selector.sample_indices_
            X1,y1 = X[idx], y[idx]
            X_selected.extend(X1)
            y_selected.extend(y1)
        return (X_selected,y_selected)
    
    def __str__(self) -> str:
        return f"{self._prototype_n_per_class}"
        # return f"{self._max_runs};{self._step_size};{self._prototype_n_per_class}"

    def _validate_params(self):
        """Validate types and values of constructor parameters.

        The expected type and values must be defined in the `_parameter_constraints`
        class attribute, which is a dictionary `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for a description of the
        accepted constraints.
        """
        pass