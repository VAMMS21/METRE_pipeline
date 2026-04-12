#!/bin/bash
#SBATCH --job-name=metre_tcn_mort
#SBATCH --output=logs/metre_tcn_mort_%j.out
#SBATCH --error=logs/metre_tcn_mort_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate metre_env

cd ~/METRE_pipeline/training

python main.py \
  --dataset_path /net/sharedfolders/datasets/MOTION/mscMEDICU/output/MIMIC_compile.npy \
  --dataset_path_cv /net/sharedfolders/datasets/MOTION/mscMEDICU/output/eICU_compile.npy \
  --model_name TCN \
  --num_channels 256 256 256 256 \
  --thresh 48 \
  --target_index 0 \
  --gap 6 \
  --epochs 150 \
  --bs 16

echo "Job finished: $(date)"
