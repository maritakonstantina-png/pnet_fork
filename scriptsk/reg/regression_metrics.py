import os
import sys
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

if len(sys.argv) > 1:
    # If the user passed a specific folder name or path via command line
    provided_path = sys.argv[1]
    if os.path.isabs(provided_path):
        metrics_dir = provided_path
    else:
        metrics_dir = os.path.join(parent_dir, provided_path)
else:
    # Find all folders that start with 'script_output_'
    all_output_folders = glob.glob(f'{parent_dir}/script_output_*/')

    # Sort them so the most recent one is at the end of the list, then select it
    metrics_dir = sorted(all_output_folders)[-1]

print(f"Reading data from: {metrics_dir}")

# Create a master list to hold all the metrics
all_metrics = []
all_predictions = []

# Dynamically determine the number of folds based on prediction files found
num_folds = len(glob.glob(f'{metrics_dir}/fold_*_predictions.csv'))
print(f"Found {num_folds} folds to process.")

for fold in range(num_folds):
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

# Create a uniquely named folder based on the specific script output folder
timestamp_folder_name = os.path.basename(os.path.normpath(metrics_dir))
output_folder = f'{parent_dir}/{timestamp_folder_name}_metrics/'
os.makedirs(output_folder, exist_ok=True)

# Convert the list of metrics into a DataFrame and save it as a CSV
metrics_df = pd.DataFrame(all_metrics)

# Calculate averages for all 10 folds add sd of the metrics
avg_pearson = metrics_df['Pearson_Correlation'].mean()
avg_spearman = metrics_df['Spearman_Correlation'].mean()
avg_r2 = metrics_df['R-squared'].mean()
avg_rmse = metrics_df['RMSE'].mean()

std_pearson = metrics_df['Pearson_Correlation'].std()
std_spearman = metrics_df['Spearman_Correlation'].std()
std_r2 = metrics_df['R-squared'].std()
std_rmse = metrics_df['RMSE'].std()

print(f"\nAverage and Std Dev Metrics across {num_folds} Folds")
print(f"Pearson Correlation: {avg_pearson:.4f} ± {std_pearson:.4f}")
print(f"Spearman Correlation: {avg_spearman:.4f} ± {std_spearman:.4f}")
print(f"R-Squared: {avg_r2:.4f} ± {std_r2:.4f}")
print(f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")


# Append the averages as a new row at the bottom of the DataFrame
avg_row = pd.DataFrame({
    'Fold': ['Average'],
    'Pearson_Correlation': [avg_pearson],
    'Spearman_Correlation': [avg_spearman],
    'R-squared': [avg_r2],
    'RMSE': [avg_rmse]
})

std_row = pd.DataFrame({
    'Fold': ['Std_Dev'],
    'Pearson_Correlation': [std_pearson],
    'Spearman_Correlation': [std_spearman],
    'R-squared': [std_r2],
    'RMSE': [std_rmse]
})

metrics_df = pd.concat([metrics_df, avg_row, std_row], ignore_index=True)

metrics_df.to_csv(f'{output_folder}/per_fold_metrics.csv', index=False)
print(f"\nSaved metrics to: {output_folder}/per_fold_metrics.csv")

# Box plot of metrics across folds
plot_metrics_df = metrics_df[~metrics_df['Fold'].isin(['Average', 'Std_Dev'])].copy()

# Reshape data for seaborn boxplot
melted_metrics = pd.melt(
    plot_metrics_df,
    id_vars=['Fold'],
    value_vars=['Pearson_Correlation', 'Spearman_Correlation', 'R-squared', 'RMSE'],
    var_name='Metric',
    value_name='Value'
)

plt.figure(figsize=(10, 6))
sns.boxplot(data=melted_metrics, x='Metric', y='Value')
sns.stripplot(data=melted_metrics, x='Metric', y='Value', color='black', alpha=0.5, jitter=True)
plt.title(f'{immune_trait.upper()} ({score_type}): Metrics Distribution Across Folds')
plt.ylabel('Metric value')
plt.tight_layout()
plt.savefig(f'{output_folder}/metrics_boxplot.png')
plt.close()

# Combine all folds for plotting
combined_df = pd.concat(all_predictions)

# Boxplot of predictions distribution per fold
plt.figure(figsize=(12, 6))
# Create a 'Fold' column in combined_df for seaborn (re-indexing or extracting from lists could work, but passing 'Fold' properly is safer)
fold_labels = []
for fold_idx in range(num_folds):
    fold_labels.extend([f"Fold {fold_idx}"] * len(all_predictions[fold_idx]))
combined_df['Fold_Label'] = fold_labels

# Melt the dataframe to have both true values and predictions side-by-side
melted_preds = pd.melt(
    combined_df, 
    id_vars=['Fold_Label'], 
    value_vars=['y_test', 'y_pred'],
    var_name='Type', 
    value_name='Value'
)
# Make legend labels nicer
melted_preds['Type'] = melted_preds['Type'].map({'y_test': 'True Value', 'y_pred': 'Predicted Value'})

sns.boxplot(data=melted_preds, x='Fold_Label', y='Value', hue='Type')
plt.title(f'{immune_trait.upper()} ({score_type}): True vs Predicted Distribution per Fold')
plt.xlabel('Fold')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/y_pred_distribution_boxplot.png')
plt.close()

# 1. Regression lines for every fold on one plot
plt.figure(figsize=(8, 6))
palette = sns.color_palette("tab10", n_colors=len(all_predictions))

min_val = min(combined_df['y_test'].min(), combined_df['y_pred'].min())
max_val = max(combined_df['y_test'].max(), combined_df['y_pred'].max())
plot_x = [min_val, max_val]

for fold, (df, color) in enumerate(zip(all_predictions, palette)):
    
    sns.regplot(
        data=df,
        x='y_pred',
        y='y_test',
        scatter_kws={'alpha': 0.18, 's': 18, 'color': color},
        line_kws={'color': color, 'linewidth': 2},
        ci=None,
        label=f'Fold {fold}'
    )

# Plot diagonal line for reference of "perfect prediction"
plt.plot(plot_x, plot_x, color='orange', linestyle='--', linewidth=2, label='y_test=y_pred')

plt.title(f"{immune_trait.upper()} ({score_type}): Regression Lines by Fold")
plt.xlabel("Predicted Value")
plt.ylabel("True Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'{output_folder}/regression_plot_by_fold.png', bbox_inches='tight')
plt.close()

# 2. Individual regression plots for each fold
for fold, (df, color) in enumerate(zip(all_predictions, palette)):
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=df,
        x='y_pred',
        y='y_test',
        scatter_kws={'alpha': 0.5, 's': 30, 'color': color},
        line_kws={'color': color, 'linewidth': 2},
        ci=None,
        label=f'Fold {fold}'
    )
    plt.plot(plot_x, plot_x, color='orange', linestyle='--', linewidth=2, label='y_test=y_pred')
    plt.title(f"{immune_trait.upper()} ({score_type}): True vs Predicted (Fold {fold})")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_folder}/regression_plot_fold_{fold}.png', bbox_inches='tight')
    plt.close()

import ast

# Plot Train vs Test loss epochs from fold_metrics.csv
fold_metrics_path = f'{metrics_dir}/fold_metrics.csv'
if os.path.exists(fold_metrics_path):
    print("Plotting training and validation loss progress...")
    fm_df = pd.read_csv(fold_metrics_path)
    
    # 1. Combined plot
    plt.figure(figsize=(10, 6))
    for idx, row in fm_df.iterrows():
        if str(row['Fold']) in ['Average', 'Std_Dev'] or pd.isna(row.get('Train_loss_epochs')):
            continue
        
        try:
            train_loss = ast.literal_eval(str(row['Train_loss_epochs']))
            test_loss = ast.literal_eval(str(row['Test_loss_epochs']))
            
            epochs = range(1, len(train_loss) + 1)
            line, = plt.plot(epochs, train_loss, label=f'Fold {row["Fold"]} Train', alpha=0.8)
            plt.plot(epochs, test_loss, label=f'Fold {row["Fold"]} Test', linestyle='--', color=line.get_color(), alpha=0.8)
            
            # 2. Individual plot per fold
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, train_loss, label='Train Loss', color='blue')
            ax.plot(epochs, test_loss, label='Test Loss', linestyle='--', color='red')
            ax.set_title(f"{immune_trait.upper()} ({score_type}): Loss Progress - Fold {row['Fold']}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            plt.tight_layout()
            fig.savefig(f'{output_folder}/loss_progress_fold_{row["Fold"]}.png', bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Could not parse train/test loss for fold {row['Fold']}: {e}")
            
    plt.title(f"{immune_trait.upper()} ({score_type}): Training & Validation Loss Progress (All Folds)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/loss_progress.png', bbox_inches='tight')
    plt.close()
    print(f"Saved loss progress plot to: {output_folder}/loss_progress.png")

