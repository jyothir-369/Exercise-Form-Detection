import mlflow

class MLFlowLogger:
    """
    MLflow experiment logger for the exercise form detection project.
    """

    def __init__(self, experiment_name="exercise_form_detection"):
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()

    def log_params(self, params: dict):
        """
        Log hyperparameters or rule thresholds.
        """
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metrics(self, metrics: dict):
        """
        Log numeric feedback such as angles over time.
        """
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    def log_artifact(self, file_path):
        """
        Log video or images (e.g., overlay result).
        """
        mlflow.log_artifact(file_path)

    def end(self):
        mlflow.end_run()
