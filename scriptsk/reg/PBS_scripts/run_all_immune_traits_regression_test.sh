#!/bin/bash

# Wrapper to run a config-only test for regression_immune_traits.py locally
# Adjust variables below as needed.

export immune_trait="nk_cells"
export score_type="PLL"

# Optional: extra hydra overrides (space-separated), then base64-encode
# Example: extra_overrides="+experiment=traits_exp1"
extra_overrides="+experiment=traits_exp1"

# Encode hydra overrides for safe passing through PBS
hydra_args="$extra_overrides"

python /shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/regression_immune_traits.py \
	--cfg job \
	parameters.trait=$immune_trait \
	parameters.score_type=$score_type \
	runtime.output_dir=/tmp/pnet_test_output \
	$hydra_args
