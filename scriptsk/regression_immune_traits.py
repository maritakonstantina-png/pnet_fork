import sys
import os
import pandas as pd
import numpy as np
#import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


sys.path.append(os.path.dirname(__file__)+"/../src/")
from pnet import Pnet
from util import util

#set arguments 
trait = sys.argv[1]
score_type = sys.argv[2]

#load data
input_dir = "/shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/aggregated_scores"
immune_trait_path = "/shares/CIBIO-Storage/BCG/scratch/kmarita/rstudio/long_matched_immune_traits_EU.csv"
output_dir = sys.argv[3]

#create input for the model, the aggregated scores for each gene (max, min, avg, delta, sd), divided by hap1 and hap2 
genetic_data = {}
for agg_func in ["avg", "sd", "max", "min", "delta"]:
    scores_hap1, scores_hap2 = util.load_hap_scores(f"{input_dir}/{score_type}/{agg_func}/", agg_func, score_type)
    genetic_data[f"{agg_func}_hap1"] = scores_hap1
    genetic_data[f"{agg_func}_hap2"] = scores_hap2

#create output, its the traits matrix and we set as index the patient_id
df_traits = pd.read_csv(immune_trait_path).set_index("tcga_patient_id")
#i need to refer to one column (one traits) per time 
immune_trait = df_traits[[trait]].apply(pd.to_numeric, errors='coerce')

#cross-validation setup 
samples = np.array(immune_trait.index.tolist())
n_splits = 5 
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

results_accumulator = {
    'gene_importances': [],
    'y_test':[],
    'y_pred' :[]
}

for fold, (train_index, test_index) in enumerate(kf.split(samples)):
    train_sample = samples[train_index].tolist()
    test_sample = samples[test_index].tolist()
        
    model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(
        genetic_data, 
        immune_trait, 
        seed=fold,           # setting seed per fold so every fold has a unique seed (random initialization, data shuffling etc.)
        dropout=0.2, 
        lr=1e-4,             # lower LR for regression, more frequent evaluation 
        weight_decay=1e-3,
        batch_size=64,      
        epochs=3000, 
        early_stopping=True, 
        train_inds=train_sample,
        test_inds=test_sample, 
        input_dropout=0.5
    )
    
    # Evaluation
    plt.clf()
    fold_results = Pnet.evaluate_and_interpret(
        model,
        test_dataset,
        immune_trait.columns.values
    )

    y_test_fold = np.array(fold_results.get('y_test')).flatten()
    y_pred_fold = np.array(fold_results.get('y_pred')).flatten()
    
    results_accumulator['y_test'].extend(y_test_fold.tolist())
    results_accumulator['y_pred'].extend(y_pred_fold.tolist())
    results_accumulator.get('gene_importances').append(fold_results.get('gene_importances'))

y_test_final = np.array(results_accumulator['y_test'])
y_pred_final = np.array(results_accumulator['y_pred'])

#calculate regression metrics
overall_mse = mean_squared_error(y_test_final, y_pred_final)
overall_r2 = r2_score(y_test_final, y_pred_final)
overall_pearson = np.corrcoef(y_test_final.flatten(), y_pred_final.flatten())[0, 1]

print(f"Final Results: MSE: {overall_mse:.4f}, R2: {overall_r2:.4f}, Pearson: {overall_pearson:.4f}")


#save metrics
pd.DataFrame({
    'metric': ['mse', 'r2', 'pearson'],
    'value': [overall_mse, overall_r2, overall_pearson]
}).to_csv(f"{output_dir}/regression_metrics.csv", index=False)

#save true and pred
pd.DataFrame({
    'y_test': results_accumulator['y_test'], 
    'y_pred': results_accumulator['y_pred']
}).to_csv(f"{output_dir}/fold_predictions.csv", index=False)

#plot true vs pred
df = pd.DataFrame({
    'y_test': results_accumulator['y_test'],
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