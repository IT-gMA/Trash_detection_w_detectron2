#!/bin/bash  --login

#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBTACH --account=interns2021026
#SBATCH --output=download_script.out
#SBATCH --export=NONE

echo "download has started"
curl -L "https://app.roboflow.com/ds/M5kDR97j43?key=XVkn79BnRY" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

