import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# Load the model and processor
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

# Updated prompt
prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Detect the language and recognize the speech:"

# Load audio from local file
audio_path = "/mnt/hdd/data/stop/test_1/alarm_test/00006152.wav"  # <— Replace this with your local file path
audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

# Prepare input for the model
inputs = processor(text=prompt, audios=audio, sampling_rate=16000, return_tensors="pt")

# Generate output
generated_ids = model.generate(**inputs, max_length=256)
generated_ids = generated_ids[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(response)
