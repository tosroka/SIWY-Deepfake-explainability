#!/bin/bash -l
#SBATCH -J eksperyment
#SBATCH -N 1
#SBATCH --output=logs/eksperyment/%x_%j.out
#SBATCH --error=logs/eksperyment/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH -p plgrid-gpu-a100
#SBATCH -A plgmusicxai01-gpu-a100
echo "Job started on $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running file $1"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
conda deactivate
source /net/pr2/projects/plgrid/plggailpwmm/tsroka/.venv310/bin/activate
module load GCC/10.3.0
module load GCC/12.3.0
module load SoX/14.4.2
module load FFmpeg/6.0
module load NVHPC/24.5-CUDA-12.4.0
cd $SLURM_SUBMIT_DIR
sh $1
echo "Job finished on $(date)"
 
