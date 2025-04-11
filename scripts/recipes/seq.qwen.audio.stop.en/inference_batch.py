import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import json
from tqdm import tqdm
import re
import jiwer


# Load the model and processor
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

# Updated prompt
prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Detect the language and recognize the speech:"



with open("scripts/recipes/seq.qwen.audio.stop.en/STOP_random5000_converted.jsonl", "r") as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

NUM_BEAMS = 5

for d in tqdm(data):

    # Load audio from local file
    audio = d["audio"]
    audio_path = f"/mnt/hdd/data/stop/{audio}"  # <— Replace this with your local file path
    
    audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

    # Prepare input for the model
    inputs = processor(text=prompt, audios=audio, sampling_rate=16000, return_tensors="pt")


    # Generate output
    generated_ids = model.generate(**inputs, max_length=256, num_beams=NUM_BEAMS, num_return_sequences=NUM_BEAMS)

    

    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    predictions = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    predictions = [p.strip() for p in predictions]
    predictions = [p.lower() for p in predictions]
    # remove special chars
    predictions = [re.sub(r"[^\w\s']", "", p) for p in predictions]


    WER = jiwer.wer(d["text"], predictions[0])
    d["wer"] = WER
    d["best_hyp"] = predictions[0]
    d["other_hyps"] = predictions[1:]

    
    
