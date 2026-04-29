import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error

immune_trait = os.environ['immune_trait']
score_type = os.environ['score_type']

# Point to the parent directory containing all the timestamped folders
parent_dir = f'/shares/CIBIO-Storage/BCG/scratch/kmarita/data/pnet_fork/{immune_trait}/{score_type}/'

# Find all folders that start with 'script_output_'
all_output_folders = glob.glob(f'{parent_dir}/script_output_*/')

# Sort them so the most recent one is at the end of the list, then select it
metrics_dir = sorted(all_output_folders)[-1]

print(f"Reading data from: {metrics_dir}")

# Create a master list to hold all the metrics
all_metrics = []
all_predictions = []

for fold in range(10):
    df = pd.read_csv(f'{metrics_dir}/fold_{fold}_predictions.csv')
    all_predictions.append(df)
    
    y_test = df['y_test']
    y_pred = df['y_pred']

    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    
    # Save the metrics for the current fold into the list
    all_metrics.append({
        'Fold': fold,
        'Pearson_Correlation': pearson_corr,
        'Spearman_Correlation': spearman_corr,
        'R-squared': r2,
        'RMSE': rmse
    })
    
    print(f"Fold {fold}:")
    print(f"Pearson Correlation: {pearson_corr}")
    print(f"Spearman Correlation: {spearman_corr}")
    print(f"R- Squared: {r2}")
    print(f"RMSE: {rmse}")

# Create a folder inside the parent directory named after the trait and score type
output_folder = f'{parent_dir}/{immune_trait}_{score_type}_metrics/'
os.makedirs(output_folder, exist_ok=True)

# Convert the list of metrics into a DataFrame and save it as a CSV
metrics_df = pd.DataFrame(all_metrics)

# Calculate averages for all 10 folds
avg_pearson = metrics_df['Pearson_Correlation'].mean()
avg_spearman = metrics_df['Spearman_Correlation'].mean()
avg_r2 = metrics_df['R-squared'].mean()
avg_rmse = metrics_df['RMSE'].mean()

print("\nAverage Metrics across 10 Folds")
print(f"Average Pearson Correlation: {avg_pearson:.4f}")
print(f"Average Spearman Correlation: {avg_spearman:.4f}")
print(f"Average R-Squared: {avg_r2:.4f}")
print(f"Average RMSE: {avg_rmse:.4f}")


# Append the averages as a new row at the bottom of the DataFrame
avg_row = pd.DataFrame({
    'Fold': ['Average'],
    'Pearson_Correlation': [avg_pearson],
    'Spearman_Correlation': [avg_spearman],
    'R-squared': [avg_r2],
    'RMSE': [avg_rmse]
})
metrics_df = pd.concat([metrics_df, avg_row], ignore_index=True)

metrics_df.to_csv(f'{output_folder}/per_fold_metrics.csv', index=False)
print(f"\nSaved metrics to: {output_folder}/per_fold_metrics.csv")

# Combine all folds for plotting
combined_df = pd.concat(all_predictions)

# 1. Regplot: True vs Predicted values
plt.figure(figsize=(8, 6))
sns.regplot(data=combined_df, x='y_pred', y='y_test', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
# Plot diagonal line for reference of "perfect prediction"
min_val = min(combined_df['y_test'].min(), combined_df['y_pred'].min())
max_val = max(combined_df['y_test'].max(), combined_df['y_pred'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='orange', linestyle='--', label='Perfect Prediction')

plt.title(f"{immune_trait.upper()} ({score_type}): True vs Predicted across 10 Folds")
plt.xlabel("Predicted Value")
plt.ylabel("True Value")
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_folder}/regression_plot.png')
plt.close()

# 2. Residuals Plot
combined_df['residuals'] = combined_df['y_test'] - combined_df['y_pred']
plt.figure(figsize=(8, 6))
sns.scatterplot(data=combined_df, x='y_pred', y='residuals', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title(f"{immune_trait.upper()} ({score_type}): Residuals vs Predicted")
plt.xlabel("Predicted Value")
plt.ylabel("Residuals (True - Predicted)")
plt.tight_layout()
plt.savefig(f'{output_folder}/residuals_plot.png')
plt.close()

# 3. Distribution Density Overlap
plt.figure(figsize=(8, 6))
sns.kdeplot(combined_df['y_test'], label='True Values', shade=True)
sns.kdeplot(combined_df['y_pred'], label='Predicted Values', shade=True)
plt.title(f"{immune_trait.upper()} ({score_type}): Distributions of True vs Predicted")
plt.xlabel("Immune Trait Score")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_folder}/distribution_plot.png')
plt.close()

print(f"Saved 3 plots in {output_folder}")