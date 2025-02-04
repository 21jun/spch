PYTHONPATH=. uv run accelerate launch --num_processes $1 scripts/recipes/seq.whisper.stop.en/inference.py --config-path conf --config-name eval-stop-seq-whisper \
model=whisper-large-v3-turbo \
data=stop_original_infer_train \
++train.per_device_train_batch_size=1 \
++train.per_device_eval_batch_size=4 
