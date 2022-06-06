#!/bin/bash
#SBATCH --chdir /home/troiani/
#SBATCH --nodes 1
#SBATCH --tasks-per-node 36
#SBATCH --mem 120G
#SBATCH --time 24:00:00
#SBATCH -o robust_linear_regression/output2L2.out
#SBATCH -e robust_linear_regression/error2L2.out
#SBATCH --partition=parallel

module load gcc
module load mvapich2
module load python

source venv/updated-venv/bin/activate

cd robust_linear_regression

srun python optimal_experiments_L2_decorrerlated_2.py

deactivate