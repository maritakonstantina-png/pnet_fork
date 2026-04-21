import sys
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


sys.path.append(os.path.dirname(__file__)+"/../src/")
from pnet import Pnet
from util import util

immune_trait = sys.argv[1]
score_type = sys.argv[2]

input_dir = "/shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/aggregated_scores"
immune_trait_path = "/shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/long_matched_immune_traits_EU.csv"
output_dir = "/shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/output_pnet"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Check if checkpoint exists
checkpoint_path = f"{output_dir}/data.pickle"
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    genetic_data = checkpoint['genetic_data']
    immune_traits = checkpoint['immune_traits']
    print("Loaded input data and target from checkpoint.")
else:
    genetic_data = {}
    for agg_func in ["avg", "sd", "max", "min", "delta"]:
        scores_hap1, scores_hap2 = util.load_hap_scores(f"{input_dir}/{score_type}/{agg_func}/", agg_func, score_type)
        genetic_data[f"{agg_func}_hap1"] = scores_hap1
        genetic_data[f"{agg_func}_hap2"] = scores_hap2

    immune_traits = pd.read_csv(immune_trait_path).dropna().set_index("tcga_patient_id")
    immune_traits = immune_traits.apply(pd.to_numeric, errors='coerce').dropna()

# Save data checkpoint
    with open(f"{output_dir}/data.pickle", "wb") as f:
     pickle.dump({'genetic_data': genetic_data, 'immune_traits': immune_traits}, f)

# Cross-Validation Setup
samples = np.array(immune_traits.index.tolist())
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

results_accumulator = {
    'gene_importances': [],
    'y_true': [],
    'y_pred': []
}

# Train
for fold, (train_index, test_index) in enumerate(kf.split(samples)):
    train_sample = samples[train_index].tolist()
    test_sample = samples[test_index].tolist()
       
    model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(
        genetic_data, 
        immune_traits, 
        seed=fold,           # setting seed per fold so every fold has a unique seed (random initialization, data shuffling etc.)
        dropout=0.2, 
        lr=1e-4,             #  lower LR for regression, more frequent evaluation 
        weight_decay=1e-3,
        batch_size=64,      
        epochs=3000, 
        early_stopping=True, 
        train_inds=train_sample,
        test_inds=test_sample, 
        input_dropout=0.5, 
        task_type="regression"
    )

    # Evaluation
    plt.clf()
    fold_results = Pnet.evaluate_and_interpret(
        model,
        test_dataset,
        immune_traits.columns.values
    )
    
    # results
    results_accumulator['y_true'].extend(fold_results['y_true'])
    results_accumulator['y_pred'].extend(fold_results['pred_proba']) # pred_proba is the predicted values for regression
    results_accumulator['gene_importances'].append(fold_results['gene_importances'])

# Metrics: convert to arrays
y_true = np.array(results_accumulator['y_true'])
y_pred = np.array(results_accumulator['y_pred'])

# Calculate Regression Metrics
overall_mse = mean_squared_error(y_true, y_pred)
overall_r2 = r2_score(y_true, y_pred)
overall_pearson = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

print(f"Final Results: MSE: {overall_mse:.4f}, R2: {overall_r2:.4f}, Pearson: {overall_pearson:.4f}")

# results
pd.DataFrame({
    'metric': ['mse', 'r2', 'pearson'],
    'value': [overall_mse, overall_r2, overall_pearson]
}).to_csv(f"{output_dir}/regression_metrics.csv", index=False)

with open(f"{output_dir}/fold_results.pickle", "wb") as f:
    pickle.dump(results_accumulator, f)

# Plotting: True vs Predicted
df = pd.DataFrame({
    'y_test': results_accumulator['y_true'],
    'y_pred': results_accumulator['y_pred']
})
sns.regplot(data=df, x='y_test', y='y_pred', color='#41B6E6')
correlation_coefficient = round(df['y_test'].corr(df['y_pred']), 2)
plt.text(0.95, 0.05, f'Correlation: {correlation_coefficient}', ha='right', va='center', transform=plt.gca().transAxes)
plt.plot([df['y_test'].min(), df['y_test'].max()], [df['y_test'].min(), df['y_test'].max()], 
         color='#FFA300', linestyle='--', label='Diagonal Line')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
sns.despine()
plt.legend()
plt.savefig(fname=f"{output_dir}/test_pred_corr.pdf", format='pdf', dpi=500)