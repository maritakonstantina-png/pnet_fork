import os
import time

import sys
import base64

pathways = [
  "HIPPO"]

score_types = ["PLL"]#,"PLLR"]

for pathway in pathways: 
    for score in score_types:
        # Collect all extra arguments passed to this script into a single string
        # Wrap the result in single quotes to escape special characters/spaces for bash safely
        extra_args = " ".join(sys.argv[1:])
        b64_args = base64.b64encode(extra_args.encode('utf-8')).decode('utf-8')

        # Setup log dir
        LOGDIR='/shares/CIBIO-Storage/BCG/scratch/kmarita/data/pnet_fork/bc/'+pathway+'/'+score+'/log/'
        os.system("mkdir -p "+ LOGDIR)
        # Launch pbs scripts
        command = "qsub \
                    -o "+LOGDIR+" \
                    -e "+LOGDIR+" \
                    -v score_type=\""+score+"\",pathway=\""+pathway+"\",hydra_b64=\""+b64_args+"\" \
                    /shares/CIBIO-Storage/BCG/scratch/kmarita/code/pnet_fork/scriptsk/bc/PBS_scripts/all_pathways.sh"
        print(command)
        os.system(command)