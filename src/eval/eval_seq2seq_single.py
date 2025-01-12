import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration



# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained(f"openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-tiny")


audio_path = "/mnt/hdd/data/librispeech/LibriSpeech/dev-clean/5338/24615/5338-24615-0000.flac"

# Select an audio file:
waveform, sampling_rate = torchaudio.load(audio_path)

# Use the model and processor to transcribe the audio:
input_features = processor(
    waveform.squeeze().numpy(), sampling_rate=sampling_rate, return_tensors="pt"
).input_features


print("input_features")
print(input_features.size())
# Generate token ids
predicted_ids = model.generate(input_features)

# Decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)