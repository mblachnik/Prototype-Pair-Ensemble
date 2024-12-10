import numpy as np
from collections import OrderedDict
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing
from imblearn.under_sampling.base import BaseUnderSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import _random_state_docstring
from sklvq import GLVQ


VOTING_KIND = ("auto", "hard", "soft")
@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class lvqClusters(BaseUnderSampler):
    _model:GLVQ
    _max_runs:int
    _step_size:float
    _batch_size:int
    _prototype_n_per_class:int
    
    
    def __init__(self, max_runs:int, step_size:float, batch_size:int, prototype_n_per_class:int, sampling_strategy="auto"):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = None
        self.estimator = None
        self.voting = "auto"
        # self._sampling_type = "bypass"
        self._max_runs = max_runs
        self._step_size = step_size
        self._batch_size = batch_size
        self._prototype_n_per_class = prototype_n_per_class
        self._model = GLVQ(
            distance_type="squared-euclidean",
            activation_type="sigmoid",
            activation_params={"beta": 1},
            solver_type="steepest-gradient-descent",
            solver_params={"max_runs": self._max_runs, "step_size": self._step_size, "batch_size": self._batch_size},
            random_state= 42,
            prototype_n_per_class = prototype_n_per_class
        )
    
    def _fit_resample(self, X, y):
        self._model.fit(X,y)
        # self.sampling_strategy_ = OrderedDict()
        return self._model.get_prototypes(), self._model.prototypes_labels_
    
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