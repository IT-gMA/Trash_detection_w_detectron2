#!/bin/bash  --login

#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBTACH --account=interns2021026
#SBATCH --output=log-install-detectron2.out

module load singularity

SINGULARITYENV_CUDA=$CUDA_HOME
time srun singularity exec --nv $MYGROUP/Detectron2_trash_recogntion/detectron2_run_container.sif python3 -m pip install -e detectron2