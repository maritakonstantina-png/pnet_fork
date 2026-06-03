import sys
import os
import tempfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow 
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, matthews_corrcoef


import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__) + "/../../src/")
from pnet import Pnet
from util import util
from mlflow_utils import search_existing_runs


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Runtime paths
    input_dir = cfg.runtime.input_dir
    phenotypes_path = cfg.runtime.phenotypes_path
    output_dir = cfg.runtime.output_dir
    id_column = cfg.runtime.get("id_column", "tcga_patient_id")

    mlflowdb_uri = cfg.runtime.mlflowdb_uri
    mlflowdb_artifact = cfg.runtime.mlflowdb_artifact

    # Parameters
    params = dict(cfg.parameters)
    score_type = params.get("score_type", "PLL")
    pathway = params.get("pathway", None)
    run_all_pathways = params.get("run_all_pathways", False)
    n_splits = params.get("n_splits", 5)
    random_state = params.get("random_state", 42)
    loss_fn_str = params.get("loss_fn", "default")
    
    if loss_fn_str.lower() in ["default"]:
        loss_fn = None
    elif loss_fn_str.lower() in ["bce"]:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = None

    mlflow.set_tracking_uri(mlflowdb_uri)

    experiment_name = "phenotypes_exp"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=mlflowdb_artifact)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)

    found_runs = search_existing_runs(params, experiment_ids=[experiment_id])
    if not found_runs:
        # Load genetic data
        genetic_data = {}
        for agg_func in ["avg", "sd", "max", "min", "delta"]:
            scores_hap1, scores_hap2 = util.load_hap_scores(
                f"{input_dir}/{score_type}/{agg_func}/", agg_func, score_type
            )
            genetic_data[f"{agg_func}_hap1"] = scores_hap1
            genetic_data[f"{agg_func}_hap2"] = scores_hap2

        
         #load phenos
        df_pheno = pd.read_csv(phenotypes_path, sep=r'\s+')
        if id_column in df_pheno.columns:
            df_pheno = df_pheno.set_index(id_column)
        
        # keep first 15 chars to match the ids of the target
        df_pheno.index = df_pheno.index.str[:15]
        
        df_pheno = df_pheno.apply(pd.to_numeric, errors="coerce")
        
        # also keep the first 15 chars from the input as well
        for key in genetic_data:
            genetic_data[key].index = genetic_data[key].index.str[:15]
            # keep the first sample that pops up for every patient if there are multiple
            genetic_data[key] = genetic_data[key][~genetic_data[key].index.duplicated(keep='first')]

        if pathway:
            pathways = [pathway]
        elif run_all_pathways:
            pathways = list(df_pheno.columns)
        else:
            raise ValueError("Set parameters.pathway or parameters.run_all_pathways=True")

        for pathway in pathways:
            y_df = df_pheno[[pathway]].dropna()

            # keep ids that exist in both input and target
            common_samples = y_df.index.intersection(genetic_data[list(genetic_data.keys())[0]].index)
            y_df = y_df.loc[common_samples]

            genetic_data_filtered = {}
            for key, data in genetic_data.items():
                genetic_data_filtered[key] = data.loc[common_samples]

            samples = np.array(common_samples.tolist())
            labels = y_df.loc[samples].values.ravel()

            # calculate baseline prevalence, positive class
            baseline_prevalence = np.sum(labels) / len(labels)

            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            pathway_output_dir = os.path.join(output_dir, pathway)
            os.makedirs(pathway_output_dir, exist_ok=True)

            run_name = f"{pathway}_{score_type}"

            all_gene_feature_importances = []
            all_additional_feature_importances = []
            all_gene_importances = []
            all_layer_importance_scores = []
            all_y_true = []
            all_pred_proba = []
            all_metrics = []

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({**params, "pathway": pathway, "score_type": score_type})

                for fold, (train_index, test_index) in enumerate(kf.split(samples, labels)):
                    train_sample = samples[train_index].tolist()
                    test_sample = samples[test_index].tolist()

                    model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(
                        genetic_data_filtered,
                        y_df,
                        seed=params.get("seed", 42),
                        dropout=params.get("dropout", 0.3),
                        lr=params.get("lr", 1e-4),
                        weight_decay=params.get("weight_decay", 1e-4),
                        batch_size=params.get("batch_size", 128),
                        epochs=params.get("epochs", 3000),
                        early_stopping=params.get("early_stopping", True),
                        train_inds=train_sample,
                        test_inds=test_sample,
                        input_dropout=params.get("input_dropout", 0.3),
                        task="BC",
                        loss_fn=loss_fn,
                    )

                    
                    # To prevent util.py list string concatenation error, pass target name as string if needed
                    results = Pnet.evaluate_and_interpret(model, test_dataset, pathway)

                    # Save loss curves for every fold 
                    util.draw_loss(train_scores, test_scores, save=os.path.join(pathway_output_dir, f"fold_{fold}_loss_curves.pdf"))

                    y_true = np.asarray(results["y_true"]).ravel()
                    pred_proba = np.asarray(results["pred_proba"]).ravel()

                    # Metrics
                    roc_auc = roc_auc_score(y_true, pred_proba)
                    prc_auc = average_precision_score(y_true, pred_proba)
                    pred_class = (pred_proba >= 0.5).astype(int)
                    acc = accuracy_score(y_true, pred_class)
                    balanced_acc = balanced_accuracy_score(y_true, pred_class)
                    mcc = matthews_corrcoef(y_true, pred_class)
                    
                    # AUCPR metrics
                    aucpr_normalized = prc_auc / baseline_prevalence if baseline_prevalence > 0 else 0
                    aucpr_adjusted = (prc_auc - baseline_prevalence) / (1 - baseline_prevalence) if baseline_prevalence < 1 else 0

                    all_metrics.append(
                        {
                            "Fold": fold,
                            "ROC_AUC": roc_auc,
                            "PRC_AUC": prc_auc,
                            "Accuracy": acc,
                            "Balanced_Accuracy": balanced_acc,
                            "MCC": mcc,
                            "AUCPR_Normalized": aucpr_normalized,
                            "AUCPR_Adjusted": aucpr_adjusted,
                            "Train_loss_epochs": train_scores,
                            "Test_loss_epochs": test_scores,
                        }
                    )

                    # Log fold metrics
                    mlflow.log_metrics(
                        {
                            f"fold_{fold}_roc_auc": roc_auc,
                            f"fold_{fold}_prc_auc": prc_auc,
                            f"fold_{fold}_accuracy": acc,
                            f"fold_{fold}_balanced_accuracy": balanced_acc,
                            f"fold_{fold}_mcc": mcc,
                            f"fold_{fold}_aucpr_normalized": aucpr_normalized,
                            f"fold_{fold}_aucpr_adjusted": aucpr_adjusted,
                        }
                    )

                    # Save predictions
                    df_pred = pd.DataFrame(index=test_dataset.input_df.index)
                    df_pred["y_test"] = y_true
                    df_pred["y_pred_proba"] = pred_proba
                    df_pred.to_csv(os.path.join(pathway_output_dir, f"fold_{fold}_predictions.csv"))

                    # Save importances per fold
                    results["gene_feature_importances"].to_csv(
                        os.path.join(pathway_output_dir, f"fold_{fold}_gene_feature_importances.csv")
                    )
                    results["additional_feature_importances"].to_csv(
                        os.path.join(pathway_output_dir, f"fold_{fold}_additional_feature_importances.csv")
                    )
                    results["gene_importances"].to_csv(
                        os.path.join(pathway_output_dir, f"fold_{fold}_gene_importances.csv")
                    )

                    all_gene_feature_importances.append(results["gene_feature_importances"])
                    all_additional_feature_importances.append(results["additional_feature_importances"])
                    all_gene_importances.append(results["gene_importances"])
                    all_layer_importance_scores.append(results["layer_importance_scores"])
                    all_y_true.append(y_true)
                    all_pred_proba.append(pred_proba)
                    
                    # Log model for this fold to MLflow artifacts
                    mlflow.pytorch.log_model(model, f"fold_{fold}_model")
                    
                    # Log fold-specific artifacts
                    mlflow.log_artifact(os.path.join(pathway_output_dir, f"fold_{fold}_predictions.csv"))
                    mlflow.log_artifact(os.path.join(pathway_output_dir, f"fold_{fold}_loss_curves.pdf"))
                    mlflow.log_artifact(os.path.join(pathway_output_dir, f"fold_{fold}_gene_feature_importances.csv"))
                    mlflow.log_artifact(os.path.join(pathway_output_dir, f"fold_{fold}_additional_feature_importances.csv"))
                    mlflow.log_artifact(os.path.join(pathway_output_dir, f"fold_{fold}_gene_importances.csv"))

                # Aggregate metrics
                metrics_df = pd.DataFrame(all_metrics)
                avg_row = pd.DataFrame(
                    {
                        "Fold": ["Average"],
                        "ROC_AUC": [metrics_df["ROC_AUC"].mean()],
                        "PRC_AUC": [metrics_df["PRC_AUC"].mean()],
                        "Accuracy": [metrics_df["Accuracy"].mean()],
                        "Balanced_Accuracy": [metrics_df["Balanced_Accuracy"].mean()],
                        "MCC": [metrics_df["MCC"].mean()],
                        "AUCPR_Normalized": [metrics_df["AUCPR_Normalized"].mean()],
                        "AUCPR_Adjusted": [metrics_df["AUCPR_Adjusted"].mean()],
                    }
                )
                metrics_df = pd.concat([metrics_df, avg_row], ignore_index=True)
                metrics_df.to_csv(os.path.join(pathway_output_dir, "fold_metrics.csv"), index=False)
                
                # Log metrics CSV to MLflow
                mlflow.log_artifact(os.path.join(pathway_output_dir, "fold_metrics.csv"))

                # Log averages to MLflow
                mlflow.log_metrics(
                    {
                        "avg_roc_auc": avg_row["ROC_AUC"].iloc[0],
                        "avg_prc_auc": avg_row["PRC_AUC"].iloc[0],
                        "avg_accuracy": avg_row["Accuracy"].iloc[0],
                        "avg_balanced_accuracy": avg_row["Balanced_Accuracy"].iloc[0],
                        "avg_mcc": avg_row["MCC"].iloc[0],
                        "avg_aucpr_normalized": avg_row["AUCPR_Normalized"].iloc[0],
                        "avg_aucpr_adjusted": avg_row["AUCPR_Adjusted"].iloc[0],
                        "baseline_prevalence": baseline_prevalence,
                    }
                )

                # Average importances
                avg_gene_feature_importances = pd.concat(all_gene_feature_importances).groupby(level=0).mean()
                avg_additional_feature_importances = pd.concat(all_additional_feature_importances).groupby(level=0).mean()
                avg_gene_importances = pd.concat(all_gene_importances).groupby(level=0).mean()

                avg_gene_feature_importances.to_csv(os.path.join(pathway_output_dir, "gene_feature_importances.csv"))
                avg_additional_feature_importances.to_csv(
                    os.path.join(pathway_output_dir, "additional_feature_importances.csv")
                )
                avg_gene_importances.to_csv(os.path.join(pathway_output_dir, "gene_importances.csv"))
                
                # Log average importance CSVs to MLflow
                mlflow.log_artifact(os.path.join(pathway_output_dir, "gene_feature_importances.csv"))
                mlflow.log_artifact(os.path.join(pathway_output_dir, "additional_feature_importances.csv"))
                mlflow.log_artifact(os.path.join(pathway_output_dir, "gene_importances.csv"))

                # Plot top 20 gene importances visually
                plt.figure(figsize=(10, 6))
                if not avg_gene_importances.empty:
                    top_genes = avg_gene_importances.iloc[:, 0].sort_values(ascending=False).head(20)
                    top_genes.plot(kind='bar', color='darkcyan')
                    plt.title(f"Top 20 Gene Importances - {pathway}")
                    plt.ylabel("Importance Score")
                    plt.tight_layout()
                    top_genes_path = os.path.join(pathway_output_dir, "top_20_gene_importances.pdf")
                    plt.savefig(top_genes_path)
                    plt.close()
                    mlflow.log_artifact(top_genes_path)

                # Plot aggregated Confusion Matrix across all folds
                if all_y_true and all_pred_proba:
                    flat_y_true = np.concatenate(all_y_true)
                    flat_pred_class = (np.concatenate(all_pred_proba) >= 0.5).astype(int)
                    cm = confusion_matrix(flat_y_true, flat_pred_class)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(cmap='Blues')
                    plt.title(f"Aggregated Confusion Matrix - {pathway}")
                    cm_path = os.path.join(pathway_output_dir, "confusion_matrix.pdf")
                    plt.savefig(cm_path)
                    plt.close()
                    mlflow.log_artifact(cm_path)

                # Plot mean ROC/PRC curves
                if all_y_true and all_pred_proba:
                    # util.py expects PyTorch tensors with .cpu().numpy()
                    all_y_true_t = [torch.tensor(y) for y in all_y_true]
                    all_pred_proba_t = [torch.tensor(p) for p in all_pred_proba]
                    
                    roc_path = os.path.join(pathway_output_dir, "roc_auc_curve.pdf")
                    prc_path = os.path.join(pathway_output_dir, "prc_auc_curve.pdf")
                    
                    util.plot_mean_roc_curve(
                        all_y_true_t,
                        all_pred_proba_t,
                        pathway,  # Pass as string, not list, to avoid util.py errors
                        roc_path,
                    )
                    util.plot_mean_prc_curve(
                        all_y_true_t,
                        all_pred_proba_t,
                        prc_path,
                    )
                    
                    # Log ROC and PRC curves to MLflow
                    mlflow.log_artifact(roc_path)
                    mlflow.log_artifact(prc_path)

                # Log final model
                mlflow.pytorch.log_model(model, "model")
            else:
                print('Model not trained since it existed already\n', params)
                print('I will go on with further analyses on the first found model')



if __name__ == "__main__":
    main()