import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

def build_query_string(params):
    """
    Builds an MLflow search query string from a dictionary of parameters.
    """
    query_parts = []
    for k, v in params.items():
        # MLflow param values are stored as strings
        query_parts.append(f"params.{k} = '{v}'")
    return " and ".join(query_parts)

def search_existing_runs(params, experiment_ids=None):
    """
    Searches for existing MLflow runs matching the given parameters.
    Returns a list of matching runs or a pandas DataFrame.
    """
    query = build_query_string(params)
    client = MlflowClient()
    
    if experiment_ids is None:
        # If no experiment_ids provided, try to get the active experiment or search all
        active_exp = mlflow.active_experiment()
        if active_exp:
            experiment_ids = [active_exp.experiment_id]
        else:
            # We can get all experiment ids
            experiments = client.search_experiments()
            experiment_ids = [exp.experiment_id for exp in experiments]

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=query
    )
    
    return runs

def get_predictions_for_run(run_id, num_folds):
    """
    Retrieves the prediction artifacts for a specific run and a given number of folds.
    """
    client = MlflowClient()
    predictions = []
    for fold in range(num_folds):
        # We assume the user wants to download the artifact and read it into a pandas dataframe
        # Or you can just get the path
        try:
            local_path = client.download_artifacts(run_id, f"fold_{fold}_predictions.csv")
            df = pd.read_csv(local_path, index_col=0)
            predictions.append(df)
        except Exception as e:
            print(f"Artifact for fold {fold} not found for run {run_id}. {e}")
    return predictions
