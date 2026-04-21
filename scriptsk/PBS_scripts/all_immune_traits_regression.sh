#!/bin/bash

# RESOURCES

#PBS -l select=1:ncpus=9:ngpus=1:mem=230gb:host=hpc3-g05-n02
#PBS -l walltime=05:00:00
#PBS -q cibioGPUQ

# MODULES LOADING
module load CUDA/12.6.0

# CONDA ENV SETUP
source /shares/CIBIO-Storage/BCG/scratch/kmarita/conda/etc/profile.d/conda.sh

# Ensure the 'pnet' conda environment exists, create from pnet.yml if missing
if ! conda info --envs | awk '{print $1}' | grep -qx "pnet"; then
    echo "Conda env 'pnet' not found — creating from pnet.yml"
    conda env create -f /shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/pnet.yml -n pnet
fi

conda activate pnet

# Ensure pandas is available in the activated env; install with pip if missing
python - <<'PY'
try:
    import pandas  # noqa: F401
except Exception:
    raise SystemExit(1)
else:
    raise SystemExit(0)
PY
if [ $? -ne 0 ]; then
    echo "pandas not found in 'pnet' env — installing via pip"
    pip install --no-cache-dir pandas
fi

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
python /shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/regression_immune_traits.py

# NOTIFY END OF SCRIPT
curl -s -d "Done $PBS_JOBID regression_immune_traits.py on cibioGPUQ launched at $start, finished at $(date)" https://ntfy.sh/kmarita