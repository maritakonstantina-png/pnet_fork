import os
import time

immune_traits = [
  "NK cells"]
 # "T helper cells" ,
  #"CD8 T cells" ,
 # "Tfh cells",
  #"AMP1" ,
 # "Term cells"
 # "aDC" ,
 # "Tcm cells" ,
  #"Eosinophils",
  #"Interferon Cluster 21214954" ,
  #"GP11 Immune IFN" ,
 # "Interferon 19272155" ,
 # "Th1 cells" ,
 # "Neutrophils"  ,
 # "Th17 cells" 
 # ]

score_types = ["PLL"] #,"PLLR"]

for trait in immune_traits: 
    for score in score_types:
        # Setup log dir
        LOGDIR='/shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/logs_all_immune_traits_regression_PLL_PLLR_func/'+trait+'/'+score+'/'
        os.system("mkdir -p "+ LOGDIR)
        # Launch pbs scripts
        command = "qsub \
                    -o "+LOGDIR+" \
                    -e "+LOGDIR+" \
                    -v score_type=\""+score+"\",immune_traits=\""+trait+"\" \
                    /shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/PBS_scripts/all_cancers_prediction_bc_all_scores.sh"
        os.system(command)