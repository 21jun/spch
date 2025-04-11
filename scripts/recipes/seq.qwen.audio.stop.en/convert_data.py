import json

with open("scripts/recipes/seq.qwen.audio.stop.en/STOP_random5000.jsonl", "r") as f:
    lines = f.readlines()

data = [json.loads(line) for line in lines]


for d in data:
    d.pop("wer")
    d.pop("best_hyp")
    d.pop("other_hyps")


with open("scripts/recipes/seq.qwen.audio.stop.en/STOP_random5000_converted.jsonl", "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")
