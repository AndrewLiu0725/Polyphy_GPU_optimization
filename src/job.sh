#!/bin/bash

#SBATCH -p gtest
#SBATCH -n 1
#SBATCH -J test
#SBATCH --account=ACD109080
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1

module load intel/2018
module load nvidia/cuda/10.0

/home/ajl870725/cudaDPLB_v0.1_wip/src/dplbe

