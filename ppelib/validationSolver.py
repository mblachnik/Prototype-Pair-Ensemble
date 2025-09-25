from typing import TYPE_CHECKING
from sklearn.model_selection import train_test_split
import numpy as np
import mlflow
from sklearn.utils import shuffle
from sklvq.objectives import ObjectiveBaseClass
from sklvq.solvers import SolverBaseClass
from sklvq.solvers._base import _update_state
from sklearn.metrics import accuracy_score

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass

STATE_KEYS = ["variables", "nit", "fun", "step_size"]


class CustomSteepestGradientDescent(SolverBaseClass):
    def __init__(
        self,
        objective: ObjectiveBaseClass,
        max_runs: int = 10,
        batch_size: int = 1,
        step_size: float = 0.1,
        callback: callable = None,
        val_split: float = 0.2,
        early_stopping: bool = False,
        patience: int = 5
    ):
        super().__init__(objective)
        self.max_runs = max_runs
        self.batch_size = batch_size
        self.step_size = step_size
        self.callback = callback
        self.val_split = val_split
        self.early_stopping = early_stopping
        self.patience = patience 

    def solve(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        model: "LVQBaseClass",
    ):
        train_data, val_data, train_labels, val_labels = train_test_split(
            data, labels, test_size=self.val_split, random_state=model.random_state_
        )
        mlflow.set_tag("early_stoped",False)
        batch_size = self.batch_size
        if batch_size > train_data.shape[0]:
            raise ValueError("Provided batch_size is invalid.")
        if batch_size <= 0:
            batch_size = train_data.shape[0]

        best_val_obj = float("inf")
        no_improvement_count = 0

        for i_run in range(self.max_runs):
            shuffled_indices = shuffle(
                np.arange(train_labels.size), random_state=model.random_state_
            )

            batches = np.array_split(
                shuffled_indices,
                list(range(batch_size, train_labels.size, batch_size)),
                axis=0,
            )

            step_size = self.step_size / (1 + i_run / self.max_runs)

            for i_batch in batches:
                batch = train_data[i_batch, :]
                batch_labels = train_labels[i_batch]

                objective_gradient = self.objective.gradient(model, batch, batch_labels)
                model.mul_step_size(step_size, objective_gradient)
                model.set_variables(
                    np.subtract(
                        model.get_variables(),
                        objective_gradient,
                        out=objective_gradient,
                    )
                )

            #
            train_obj = self.objective(model, train_data, train_labels)
            # train_acc = accuracy_score(model.predict(train_data), train_labels)
            val_obj = self.objective(model, val_data, val_labels)
            val_acc = accuracy_score(model.predict(val_data), val_labels)

            mlflow.log_metric("train_objective", train_obj, step=i_run)
            # mlflow.log_metric("train_acc", train_acc, step=i_run)

            mlflow.log_metric("val_objective", val_obj, step=i_run)
            mlflow.log_metric("val_acc", val_acc, step=i_run)

            # Early stopping
            if self.early_stopping:
                if val_obj < best_val_obj - 1e-6:
                    best_val_obj = val_obj
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= self.patience:
                        mlflow.set_tag("early_stoped",True)
                        mlflow.log_metric("last_itter",i_run+1)
                        print(f"Early stopping triggered at iteration {i_run + 1}")
                        break

            # Callback
            if self.callback is not None:
                state = _update_state(
                    STATE_KEYS,
                    variables=np.copy(model.get_variables()),
                    nit=i_run + 1,
                    fun=train_obj,
                    step_size=step_size,
                )
                if self.callback(state):
                    return
