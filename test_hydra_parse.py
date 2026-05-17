import yaml
with open("/shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/traits_exp1.yaml") as f:
    d = yaml.safe_load(f)
print(d['hydra']['sweeper']['params']['loss_fn'])
