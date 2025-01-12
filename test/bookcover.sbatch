#!/bin/bash
#SBATCH -J testrun # job name
#SBATCH -o ./output_testrun.%j.out 
#SBATCH -p A5000 # queue name or partition name
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
cat test/finetune.sh
sh test/finetune.sh

curl  -d "Job finished 😀" ntfy.sh/lee1jun