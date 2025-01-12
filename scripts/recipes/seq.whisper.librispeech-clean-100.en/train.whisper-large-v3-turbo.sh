accelerate launch --num_processes $1 src/train/train_seq2seq.py --config-path ../../scripts/recipes/seq.whisper.librispeech-clean-100.en/conf --config-name librispeech-seq-whisper
