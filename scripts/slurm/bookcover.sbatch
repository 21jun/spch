#!/bin/bash
#SBATCH -J testrun # job name
#SBATCH -o ./output_testrun.%j.out 
#SBATCH -p RTX6000ADA # queue name or partition name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16



# Your commands go here
export MLFLOW_TRACKING_URI="http://newton.postech.ac.kr:8850" 
export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="1qazxc"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

export LIBRISPEECH_ROOT=/home1/lee1jun/data/librispeech
cat scripts/recipes/ctc.librispeech-clean-100.en/train.wav2vec2.sh
sh scripts/recipes/ctc.librispeech-clean-100.en/train.wav2vec2.sh 8

curl  -d "Job finished 😀" ntfy.sh/lee1jun