#!/usr/bin/env python3
import subprocess
import sys
import time
import random

# Random sleep (0-10s) to prevent two simultaneous jobs from 
# seeing the same "free" GPU.
time.sleep(random.uniform(0, 10))

try:
    # Query nvidia-smi for index and memory used
    # Returns lines like: "0, 10" (Index 0 has 10MB used)
    cmd = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
    output = subprocess.check_output(cmd, shell=True).decode('utf-8')

    lines = output.strip().split('\n')
    
    threshold_mb = 1000
    
    for line in lines:
        if not line.strip(): continue
        index, mem_used = line.split(',')
        if int(mem_used) < threshold_mb:
            # Print ONLY the index number and exit
            print(index.strip())
            sys.exit(0)
            
    # If we get here, no GPU is free.
    # Print nothing (or -1) and exit with error code
    sys.exit(1)

except Exception:
    sys.exit(1)
