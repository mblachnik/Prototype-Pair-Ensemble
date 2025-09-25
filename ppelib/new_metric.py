from sklearn.metrics import accuracy_score
import mlflow

def mlflow_scoring_wrapper(estimator, X, y):
    # Start new run inside GridSearchCV scoring
    with mlflow.active_run() as cur_run:# .start_run(nested=True):  # nested to avoid clashing
        y_pred = estimator.predict(X)
        acc = accuracy_score(y, y_pred)

        # Log params & metrics
        if hasattr(estimator, 'get_params'):
            params = estimator.get_params(deep=True)
            mlflow.log_params(params)
            try:
                mlflow.set_tag("prototype_n_per_class",params.get("prototype_n_per_class",None))
                mlflow.set_tags(params.get("solver_params"))
            except:pass
        mlflow.log_metric("accuracy", acc)
        return acc
