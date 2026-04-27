#!/bin/bash

# RESOURCES

#PBS -l select=1:ncpus=8:ngpus=1:mem=230gb
#PBS -l walltime=05:00:00
#PBS -q commonGPUQ

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

# SCRIPT
python /shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/regression_immune_traits.py $immune_trait $score_type $output_dir

# NOTIFY END OF SCRIPT
#curl -s -d "Done $PBS_JOBID regression_immune_traits.py on cibioGPUQ launched at $start, finished at $(date)" https://ntfy.sh/kmarita