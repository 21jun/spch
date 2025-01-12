import torchaudio

print(torchaudio.__version__)
torchaudio.load(
    "/mnt/hdd/data/librispeech/LibriSpeech/dev-clean/1272/128104/1272-128104-0007.flac"
)
