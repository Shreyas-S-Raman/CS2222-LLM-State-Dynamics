#!/bin/bash
#SBATCH -n 3
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 5:00:00

### modify seed below
#SBATCH --array=0-0
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-train-%A/%a.err
#SBATCH -o sbatch_out/arrayjob-train-%A/%a.out
source ~/.bashrc
source cs2222/bin/activate
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

model='gpt2-large'
echo "RUNNING ${model}"

python3 evaluate_model.py --model_name $model -r
echo "FINISHED RUNNING ${model}$"

# https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html?utm_source=chatgpt.com
# Range of values to try:
# GPT2: gpt2-small, gpt2-medium, gpt2-large, gpt2-xl, othello-gpt
# OPT: facebook/opt-125m, facebook/opt-2.7b, facebook/opt-6.7b
# TinyStories: "tiny-stories-1M", "tiny-stories-3M", tiny-stories-28M
# Pythia: pythia-14m, pythia-70m, pythia-1.4b
# LLaMa: llama-7b, llama-13b, llama-30b [DOESN'T WORK DIRECTLY]
# T5: t5-small, t5-base, t5-large
#DONE: 
#Pythia: 14m, 70m
#TinyStories: 1m, 3m, 28m
#GPT2: gpt2-small, gpt2-medium, gpt2-large, gpt2-xl
#OPT: 
