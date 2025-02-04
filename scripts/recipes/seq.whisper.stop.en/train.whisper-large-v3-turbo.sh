PYTHONPATH=. uv run accelerate launch --num_processes $1 src/train/train_seq2seq.py --config-path ../../scripts/recipes/seq.whisper.stop.en/conf --config-name train-stop-seq-whisper \
model=whisper-large-v3-turbo \
++train.per_device_train_batch_size=1 \
++train.per_device_eval_batch_size=1 \
++train.gradient_accumulation_steps=2 \
++train.learning_rate=5e-6 \
++train.num_epochs=100 \
++train.eval_steps=1