import sys
import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
import mlflow 
import time

import hydra
from omegaconf import DictConfig
import torch.nn as nn

sys.path.append(os.path.dirname(__file__)+"/../src/")
from pnet import Pnet
from util import util


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig)
    input_dir = cfg.runtime.input_dir
    phenotypes_path = cfg.runtime.phenotypes_path_path
    output_dir = cfg.runtime.output_dir
    
    mlflowdb_uri = cfg.runtime.mlflowdb_uri
    mlflowdb_artifact = cfg.runtime.mlflowdb_artifact

     mlflow.set_tracking_uri(mlflowdb_uri)
     
    experiment_name = "phenotypes_exp"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflowdb_artifact)
    mlflow.set_experiment(experiment_name)








if __name__ == "__main__":
    main()