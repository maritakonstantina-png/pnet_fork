import os
import time

immune_traits = [
 # "nk_cells",
 #"t_helper_cells" ,
 #"cd8_t_cells" ,
 #"tfh_cells",
  "apm1" ,
  "tem_cells",
  "a_dc" ],
  #"tcm_cells" ,
  #"eosinophils",
 # "interferon_cluster_21214954" ,
 # "gp11_immune_ifn" ,
 # "interferon_19272155" ,
 # "th1_cells" ,
 # "neutrophils"  ,
 # "th17_cells" 
 #]

score_types = ["PLL","PLLR"]

for trait in immune_traits: 
    for score in score_types:
        # Setup log dir
        LOGDIR='/shares/CIBIO-Storage/BCG/scratch/kmarita/data/pnet_fork/'+trait+'/'+score+'/log/'
        os.system("mkdir -p "+ LOGDIR)
        # Launch pbs scripts
        command = "qsub \
                    -o "+LOGDIR+" \
                    -e "+LOGDIR+" \
                    -v score_type=\""+score+"\",immune_trait=\""+trait+"\" \
                    /shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/PBS_scripts/all_immune_traits_regression.sh"
        os.system(command)