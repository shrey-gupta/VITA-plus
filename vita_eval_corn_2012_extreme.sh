#!/bin/bash
#SBATCH --job-name=vita-corn-2012-extreme
#SBATCH --partition=medium
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/vita_corn_2012_extreme_%j.out
#SBATCH --error=logs/vita_corn_2012_extreme_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guptsh@bc.edu

module load cuda/12.4.1_gcc11.4.1
source ~/.bashrc
conda activate shrey-virtual-hpc-conda

cd /scratch/guptsh/working-directory/vita-hpc-attn-bias/
mkdir -p logs

echo "Starting VITA fine-tuning for corn, gridMET weekly"

python run_gridmet_corn_2012.py \
  --pretrained-model-path /scratch/guptsh/working-directory/VITA-main/checkpoints/vita_sinusoid.pth \
  --batch-size 16 \
  --n-epochs 40 \
  --model-size small \
  --beta 1e-4 \
  --init-lr 1e-4 \
  --test-type extreme \
  --crop-type corn \
  --test-year 2013 \
  --n-train-years 28 \
  --n-past-years 5 \
  --weather-vars all \ 
  --cvar-frac 0.0 \
  --drift-weight-strength 0.0 \
  --feature-dropout-prob 0.0 \
  --attn-bias-strength 0.0

# Previous (non-original) behavior for reference:
python run_gridmet_corn_2012.py \
  --pretrained-model-path checkpoints/vita_sinusoi.pth \
  --n-train-years 28 \
  --n-past-years 3 \
  --test-type extreme \
  --batch-size 8 \
  --n-epochs 40 \
  --year-weights '{"1983":3, "1988":3,"1993":3,"2002":3}' \
  --cvar-frac 0.0 \
  --drift-weight-strength 0.0 \
  --drift-min-weight 0.2 \
  --drift-features vpd \
  --drift-target-year 2013 \
  --feature-dropout-prob 0.1 \
  --feature-dropout-protect 12 \
  --attn-bias-strength 0.5

echo "Running full-prediction metrics for 2013..."
PYTHONPATH=. python eval_vita_single_year.py \
  --model-path data/trained_models/crop_yield/vita_corn_yield_2.0m_best.pth \
  --crop-type corn \
  --test-year 2013 \
  --n-train-years 28 \
  --n-past-years 5 \
  --model-size small \
  --k 1 \
  --batch-size 64 \
  --test-type extreme \
  --weather-vars all \
  --plot-attention \
  --plot-attention-heatmap \
  --attention-out logs/attention_2013.png \
  --attention-csv logs/attention_2013.csv \
  --attention-heatmap-out logs/attention_heatmap_2009_2013.png \
  --save-scatter
