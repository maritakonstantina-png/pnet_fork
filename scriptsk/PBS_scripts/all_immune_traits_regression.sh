#!/bin/bash

# RESOURCES

#PBS -l select=1:ncpus=9:ngpus=1:mem=230gb
#PBS -l walltime=05:00:00
#PBS -q cibioGPUQ

# MODULES LOADING
module load CUDA/12.6.0

# CONDA ENV SETUP
source /shares/CIBIO-Storage/BCG/scratch/kmarita/conda/etc/profile.d/conda.sh

# # Ensure the 'pnet' conda environment exists, create from pnet.yml if missing
# if ! conda info --envs | awk '{print $1}' | grep -qx "pnet"; then
#     echo "Conda env 'pnet' not found — creating from pnet.yml"
#     conda env create -f /shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/pnet.yml -n pnet
# fi

conda activate pnet

# GET START TIME
start=$(date)
timestamp=$(date +"%Y%m%d_%H%M%S")

# SETUP SCRIPT OUTPUT FOLDER
output_dir='/shares/CIBIO-Storage/BCG/scratch/kmarita/data/pnet_fork/'$immune_trait'/'$score_type'/script_output_'$timestamp'/'
mkdir -p $output_dir
#set 
mlflowdb_uri='sqlite:////shares/CIBIO-Storage/BCG/scratch/kmarita/data/pnet_fork/mlflow.db'
mlflowdb_artifacts='/shares/CIBIO-Storage/BCG/scratch/kmarita/data/pnet_fork/mlruns'



# Look for a free GPU
echo "Looking for a GPU ... "
FREE_GPU=""
while [ -z "$FREE_GPU" ]; do
    FREE_GPU=$(python /shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/PBS_scripts/get_gpu.py)
    
    if [ -z "$FREE_GPU" ]; then
        echo "No GPU free yet. Waiting 3s..."
        sleep 3
    fi
done

# 2. Export the variable
export CUDA_VISIBLE_DEVICES=$FREE_GPU

echo "Assigned to GPU: $CUDA_VISIBLE_DEVICES"
 
# GET START TIME
start=$(date)

# Decode Hydra args securely to avoid any PBS syntax problems with commas
hydra_args=$(echo "$hydra_b64" | base64 --decode)

# SCRIPT
# python /shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/regression_immune_traits.py params.trait=$immune_trait params.score_type=$score_type output_dir=$output_dir $hydra_args
echo "/shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/regression_immune_traits.py -m params.trait=$immune_trait params.score_type=$score_type output_dir=$output_dir $hydra_args"
