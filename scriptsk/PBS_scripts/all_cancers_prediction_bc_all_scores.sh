#!/bin/bash

# RESOURCES

#PBS -l select=1:ncpus=9:ngpus=1:mem=230gb:host=hpc3-g05-n02
#PBS -l walltime=05:00:00
#PBS -q cibioGPUQ

# MODULES LOADING
module load CUDA/12.6.0

# CONDA ENV SETUP
source /shares/CIBIO-Storage/BCG/scratch/kmarita/conda/etc/profile.d/conda.sh
conda activate pnet

# SETUP SCRIPT OUTPUT FOLDER
mkdir -p '/shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/output_pnet'

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
python /shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/regression_immune_traits.py" >> "/shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/output_pnet/regression_immune_traits_$PBS_JOBID.log" 2>&1

# NOTIFY END OF SCRIPT
curl -d "Done $PBS_JOBID ($immune_traits) on cibioGPUQ launched at $start, finished at $(date)" ntfy.sh/kmarita
