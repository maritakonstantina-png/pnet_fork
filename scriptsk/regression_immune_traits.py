import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import time

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
immune_trait = df_traits[[trait]].apply(pd.to_numeric)

#initialize the lists you want to save later 
all_gene_importances = []
all_gene_feature_importances = []
all_additional_feature_importances = []
#all_layer_importance_scores = []
all_dfs =[] #both y_test and y_pred


#cross validation 
samples = np.array(immune_trait.index.tolist())
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state =42)


for fold, (train_index, test_index) in enumerate(kf.split(samples)):
    train_sample = samples[train_index].tolist()
    test_sample = samples[test_index].tolist()
    
    #run pnet
    model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(
        genetic_data, 
        immune_trait, 
        seed=fold,           # setting seed per fold so every fold has a unique seed (random initialization, data shuffling etc.)
        dropout=0.1, # was 0.2 before
        lr=1e-4,             # lower LR for regression, more frequent evaluation 
        weight_decay=1e-3,
        batch_size=64,      
        epochs=3000, 
        early_stopping=True, 
        train_inds=train_sample,
        test_inds=test_sample, 
        input_dropout=0.2,  #was 0.5 before
    )
    
    # move model to CPU for prediction and interpretation
    model.to('cpu')
    
    x_train = train_dataset.x
    y_train = train_dataset.y
    additional_train = train_dataset.additional  # because on genetic_data i have more than one inputs(avg, min etc.)
    x_test = test_dataset.x
    y_test = test_dataset.y
    additional_test = test_dataset.additional


    #predict. test_datset.x is assigned the first key pnet receives which is avg_hap1 and the rest 9 are assigned to test_dataset.additional
    y_pred = model.predict(test_dataset.x, test_dataset.additional).detach()
    df = pd.DataFrame(index=test_dataset.input_df.index)
    df['y_test'] = test_dataset.y
    df['y_pred'] = y_pred
    #connect the empty list to the values of y_test and y_pred
    all_dfs.append(df)
    
    # Save predictions for this specific fold
    df.to_csv(f"{output_dir}/fold_{fold}_predictions.csv")
    

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
final_predictions = pd.concat(all_dfs)
final_predictions.to_csv(f"{output_dir}/final_predictions_all_folds.csv")

