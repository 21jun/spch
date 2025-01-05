accelerate launch --num_processes $1 src/train/train_ctc.py --config-path ../../scripts/recipes/ctc.librispeech-clean-100.en/conf --config-name librispeech-ctc \
++train.per_device_train_batch_size=16 \
++train.gradient_accumulation_steps=2 \
++train.learning_rate=1e-6 \
++train.num_epochs=100 