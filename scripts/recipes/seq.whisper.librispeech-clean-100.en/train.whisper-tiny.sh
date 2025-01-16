PYTHONPATH=. uv run accelerate launch --main_process_port $((30000 + RANDOM % 20001)) --num_processes $1 src/train/train_seq2seq.py --config-path ../../scripts/recipes/seq.whisper.librispeech-clean-100.en/conf --config-name librispeech-seq-whisper \
model=whisper-tiny \
++train.per_device_train_batch_size=16 \
++train.per_device_eval_batch_size=16 \
++train.gradient_accumulation_steps=2 \
++train.learning_rate=5e-6 \
++train.num_epochs=100 \
++train.eval_steps=100