PYTHONPATH=. uv run accelerate launch --num_processes $1 src/train/train_seq2seq.py --config-path ../../scripts/recipes/seq.whisper.librispeech-clean-100.en/conf --config-name librispeech-seq-whisper \
++train.per_device_train_batch_size=32 \
++train.per_device_eval_batch_size=32 \
++train.gradient_accumulation_steps=1 \
++train.learning_rate=1e-4 \
++train.num_epochs=100 \
++train.eval_steps=10