#!/bin/bash  --login

#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBTACH --account=interns2021026
#SBATCH --output=mdoan_run2.out
#SBATCH --export=NONE

module load cuda
module load singularity

export SINGULARITYENV_PYTHONUSERBASE="/group/interns2021026/mdoan/user_python"
export SINGULARITYENV_MPLCONFIGDIR="/group/interns2021026/mdoan/user_python/matplotlib"
export SINGULARITYENV_TORCH_HOME="/group/interns2021026/mdoan/torch_home"

SINGULARITYENV_CUDA=$CUDA_HOME
time srun singularity exec --nv $MYGROUP/Detectron2_trash_recogntion/detectron2_run_container.sif python $MYGROUP/Detectron2_trash_recogntion/src/train.py --resume true
