PYTHONPATH=. uv run accelerate launch --main_process_port $((30000 + RANDOM % 20001)) --num_processes $1 src/train/train_seq2seq.py --config-path ../../scripts/recipes/seq.whisper.ksponspeech.ko/conf --config-name ksponspeech-seq-whisper \
model=whisper-tiny \
++train.per_device_train_batch_size=32 \
++train.per_device_eval_batch_size=32 \
++train.gradient_accumulation_steps=2 \
++train.learning_rate=1e-4 \
++train.num_epochs=10 \
++train.eval_steps=100