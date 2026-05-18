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
def main(cfg: DictConfig):
    # Set arguments from Hydra config
    input_dir = cfg.runtime.input_dir
    immune_trait_path = cfg.runtime.immune_trait_path
    output_dir = cfg.runtime.output_dir
    
    mlflowdb_uri = cfg.runtime.mlflowdb_uri
    mlflowdb_artifact = cfg.runtime.mlflowdb_artifact

    # Interpret the custom loss function parameter
    loss_fn_str = str(cfg.parameters.loss_fn).lower().strip()
    if "mse" in loss_fn_str:
        loss_fn = nn.MSELoss()
    elif "l1" in loss_fn_str or "mae" in loss_fn_str: #mae
        loss_fn = nn.L1Loss()
    elif "huber" in loss_fn_str:
        loss_fn = nn.HuberLoss()
    elif "pnet" in loss_fn_str:
        loss_fn = None
    else:
        raise NotImplementedError(f"Loss function {loss_fn_str} not implemented.")

    mlflow.set_tracking_uri(mlflowdb_uri)

    experiment_name = "immune_traits_exp"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflowdb_artifact)
    mlflow.set_experiment(experiment_name)

    params = dict(cfg.parameters)
    # No need to overwrite params["loss_fn"] since we are now logging standard dict(cfg.parameters)
    score_type = params['score_type']
    trait = params['trait']
    normalize = params.get('normalize', False)  # Default to False (use experiment to enable)
    #create input for the model, the aggregated scores for each gene (max, min, avg, delta, sd), divided by hap1 and hap2 
    genetic_data = {}
    for agg_func in ["avg", "sd", "max", "min", "delta"]:
        scores_hap1, scores_hap2 = util.load_hap_scores(f"{input_dir}/{score_type}/{agg_func}/", agg_func, score_type)
        genetic_data[f"{agg_func}_hap1"] = scores_hap1
        genetic_data[f"{agg_func}_hap2"] = scores_hap2

    #create output, its the traits matrix and we set as index the patient_id
    df_traits = pd.read_csv(immune_trait_path).set_index("tcga_patient_id")
    #i need to refer to one column (one traits) per time 
    immune_trait = df_traits[[trait]].apply(pd.to_numeric)

    #initialize the lists you want to save later 
    all_gene_importances = []
    all_gene_feature_importances = []
    all_additional_feature_importances = []
    #all_layer_importance_scores = []
    all_dfs =[] #both y_test and y_pred
    all_metrics = [] # to store metrics per fold


    #cross validation 
    samples = np.array(immune_trait.index.tolist())
    n_splits = params['n_splits']
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=params['random_state'])

    # Scalers for input features
    scalers_genetic_data = {}
    # Scaler for target
    target_scaler = StandardScaler()

    # Set the run name explicitly to differentiate traits and scores
    run_name = f"{trait}_{score_type}"

    # Training with MLflow logging
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)


        for fold, (train_index, test_index) in enumerate(kf.split(samples)):
            train_sample = samples[train_index].tolist()
            test_sample = samples[test_index].tolist()
            
            # Create a copy of genetic data for this fold
            genetic_data_for_training = genetic_data.copy()
            immune_trait_fold = immune_trait.copy()
            
            # Apply normalization if enabled
            if normalize:
                # Normalize genetic data
                for key in genetic_data_for_training.keys():
                    train_data = genetic_data_for_training[key].loc[train_sample].values
                    scaler = StandardScaler()
                    scaler.fit(train_data)
                    
                    # Create normalized version
                    normalized_data = genetic_data_for_training[key].copy()
                    normalized_data.loc[train_sample] = scaler.transform(train_data)
                    normalized_data.loc[test_sample] = scaler.transform(
                        genetic_data_for_training[key].loc[test_sample].values
                    )
                    genetic_data_for_training[key] = normalized_data
                    scalers_genetic_data[f"{key}_fold{fold}"] = scaler
                
                # Normalize target
                y_train = immune_trait.loc[train_sample].values.reshape(-1, 1)
                y_test = immune_trait.loc[test_sample].values.reshape(-1, 1)
                
                target_scaler_fold = StandardScaler()
                target_scaler_fold.fit(y_train)
                
                # Assign 2D arrays (n_samples, 1) into the single-column DataFrame to avoid shape mismatches
                immune_trait_fold.loc[train_sample] = target_scaler_fold.transform(y_train)
                immune_trait_fold.loc[test_sample] = target_scaler_fold.transform(y_test)
                # Plot normalized inputs and target distributions for inspection
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    keys = list(genetic_data_for_training.keys())
                    n_keys = len(keys)
                    ncols = 5
                    nrows = (n_keys + ncols - 1) // ncols
                    plt.figure(figsize=(4 * ncols, 3 * (nrows + 1)))
                    for i, key in enumerate(keys, start=1):
                        ax = plt.subplot(nrows + 1, ncols, i)
                        data_vals = genetic_data_for_training[key].values.flatten()
                        ax.hist(data_vals, bins=100, density=True, color='C0', alpha=0.7)
                        ax.set_title(key)
                        ax.set_xlim(-5, 5)
                    # Plot target on its own row
                    ax = plt.subplot(nrows + 1, 1, nrows + 1)
                    target_vals = immune_trait_fold.loc[train_sample].values.flatten()
                    ax.hist(target_vals, bins=100, density=True, color='C3', alpha=0.7)
                    ax.set_title(f"target_{trait}_fold{fold}")
                    ax.set_xlim(-5, 5)
                    plt.tight_layout()
                    plot_path = os.path.join(output_dir, f"fold_{fold}_normalization_plot.png")
                    plt.savefig(plot_path, dpi=150)
                    plt.close()
                    print("normalization plot done")
                except Exception:
                    # Don't crash the run on plotting errors
                    import traceback
                    traceback.print_exc()
            else:
                # No normalization - use original data
                target_scaler_fold = None
            
            #run pnet
            model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(
                genetic_data_for_training, 
                immune_trait_fold, 
                seed=params['seed'],     
                dropout=params['dropout'], 
                lr=params['lr'], 
                weight_decay=params['weight_decay'],
                batch_size=params['batch_size'],      
                epochs=params['epochs'], 
                early_stopping=params['early_stopping'], 
                train_inds=train_sample,
                test_inds=test_sample, 
                input_dropout=params['input_dropout'],
                loss_fn=loss_fn
            )
            
            # move model to CPU for prediction and interpretation
            model.to('cpu')
            
            x_train = train_dataset.x
            y_train_pred = train_dataset.y
            additional_train = train_dataset.additional
            x_test = test_dataset.x
            y_test_pred = test_dataset.y
            additional_test = test_dataset.additional

            print(f"Starting forward pass on the cpu for fold {fold}")
            #predict
            y_pred = model.predict(test_dataset.x, test_dataset.additional).detach()
            
            # Inverse transform if normalization was applied
            if normalize and target_scaler_fold is not None:
                y_pred = target_scaler_fold.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).flatten()
                y_test_original = target_scaler_fold.inverse_transform(test_dataset.y.reshape(-1, 1)).flatten()
            else:
                y_pred = y_pred.cpu().numpy().flatten() if hasattr(y_pred, 'cpu') else y_pred
                y_test_original = test_dataset.y
            
            df = pd.DataFrame(index=test_dataset.input_df.index)
            df['y_test'] = y_test_original
            df['y_pred'] = y_pred
            #connect the empty list to the values of y_test and y_pred
            all_dfs.append(df)
            
            # Save predictions for this specific fold
            df.to_csv(f"{output_dir}/fold_{fold}_predictions.csv")
            
            # Compute metrics for this specific fold
            pearson_corr, _ = pearsonr(df['y_test'], df['y_pred'])
            spearman_corr, _ = spearmanr(df['y_test'], df['y_pred'])
            r2 = r2_score(df['y_test'], df['y_pred'])
            rmse = mean_squared_error(df['y_test'], df['y_pred']) ** 0.5
            
            # Store metrics for this fold
            all_metrics.append({
                'Train_loss_epochs': train_scores,
                'Test_loss_epochs': test_scores,
                'Fold': fold,
                'Pearson_Correlation': pearson_corr,
                'Spearman_Correlation': spearman_corr,
                'R-squared': r2,
                'RMSE': rmse
            })

            # Log epoch-level metrics for this fold
            for epoch, (train_loss, test_loss) in enumerate(zip(train_scores, test_scores)):
                mlflow.log_metrics({
                    f"fold_{fold}_train_loss": train_loss,
                    f"fold_{fold}_test_loss": test_loss
                }, step=epoch)

            #calculates the contribution score of each gene = importance. It aggregates these scores at a feature level(the inmportance of avg_hap1)
            #then at a gene level zand then at the pathway level (the importance of the hidden layer nodes, which represent gene sets or pathways)
            #gene_feature_importances = the importance of each input(ang, min,..) for each gene 
            #additional_feature_importances = which of the 10 input data was more important 
            #gene_importance = total importance of each gene 
            #layer_importance_scores = the importance of each node in the pathway layers
            gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = model.interpret(test_dataset)
            layer_list = [gene_feature_importances, additional_feature_importances, gene_importances] + layer_importance_scores
            layer_list_names = ['gene_feature', 'additional_feature', 'gene'] + [f'layer_{i}' for i in range(5)]
            layer_list_dict = dict(zip(layer_list_names, layer_list))


            # save the importance scores for the current fold
            gene_feature_importances.to_csv(f"{output_dir}/fold_{fold}_gene_feature_importances.csv")
            additional_feature_importances.to_csv(f"{output_dir}/fold_{fold}_additional_feature_importances.csv")
            gene_importances.to_csv(f"{output_dir}/fold_{fold}_gene_importances.csv")
            #layer_importance_scores.to_csv(f"{output_dir}/fold_{fold}_layer_importances_scores.csv")

            #append results to lists so i can average them later
            all_gene_feature_importances.append(gene_feature_importances)
            all_additional_feature_importances.append(additional_feature_importances)
            all_gene_importances.append(gene_importances)
            #all_layer_importance_scores.append(layer_importance_scores) also its a list of dfs so needs diff saving

        #outside the for loop 
        #average the importances and save them 
        avg_gene_feature_importances = pd.concat(all_gene_feature_importances).groupby(level=0).mean()
        avg_additional_feature_importances = pd.concat(all_additional_feature_importances).groupby(level=0).mean()
        avg_gene_importances = pd.concat(all_gene_importances).groupby(level=0).mean()
        #avg_layer_importance_scores = pd.concat(all_layer_importance_scores).groupby(level=0).mean()

        avg_gene_feature_importances.to_csv(f"{output_dir}/gene_feature_importances.csv")
        avg_additional_feature_importances.to_csv(f"{output_dir}/additional_feature_importances.csv")
        avg_gene_importances.to_csv(f"{output_dir}/gene_importances.csv")
        #layer_importance_scores.to_csv(f"{output_dir}/layer_importances_scores.csv")

        #save y_true and y_predict
        final_predictions = pd.concat(all_dfs) # list showing how your model performed on every patient the single time it was asked to predict their score without having seen them in training
        final_predictions.to_csv(f"{output_dir}/final_predictions_all_folds.csv")

        # Ensure metrics are saved
        metrics_df = pd.DataFrame(all_metrics)

        # Calculate averages
        avg_row = pd.DataFrame({
            'Fold': ['Average'],
            'Pearson_Correlation': [metrics_df['Pearson_Correlation'].mean()],
            'Spearman_Correlation': [metrics_df['Spearman_Correlation'].mean()],
            'R-squared': [metrics_df['R-squared'].mean()],
            'RMSE': [metrics_df['RMSE'].mean()]
        })

        metrics_df = pd.concat([metrics_df, avg_row], ignore_index=True)
        metrics_df.to_csv(f"{output_dir}/fold_metrics.csv", index=False)
        
        #log the metrics using the pre-calculated averages
        mlflow.log_metrics({
            "avg_pearson": avg_row['Pearson_Correlation'][0],
            "avg_spearman": avg_row['Spearman_Correlation'][0],
            "avg_r2": avg_row['R-squared'][0],
            "avg_rmse": avg_row['RMSE'][0]
        })
        # Log final model
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()