import yaml
import json


def convert_jsonl_to_yaml(json_file, yaml_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for d in data:
        # rename the key audio_path to audio
        d["audio"] = d.pop("audio_path")

    with open(yaml_file, "w", encoding="utf-8") as f:

        
        data = {"data": data, "meta": {"audio_root": "/mnt/hdd/data/KsponSpeech_wav"}}
        yaml.dump(data, f, allow_unicode=True, width=890)


# scripts/recipes/ksponspeech.all/data/ksponspeech_dev.jsonl
ROOT = "scripts/recipes/ksponspeech.all/data/"
convert_jsonl_to_yaml(
    ROOT+"ksponspeech_eval_clean.jsonl", ROOT+"ksponspeech_eval_clean.yaml"
)
