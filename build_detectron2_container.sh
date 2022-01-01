#!/bin/bash  --login

#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBTACH --account=interns2021026

module load singularity

time srun singularity remote login --tokenfile topaz-detectron2
time srun singularity build -r detectron2_run_container.sif detectron2.def