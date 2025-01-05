import argparse
import glob
import json
import os
import wave

import tqdm
import yaml

SETS = {"train": ["train-clean-100"], "dev": ["dev-clean"], "test": ["test-clean"]}


def load_transcripts(path):
    pattern = os.path.join(path, "*/*/*.trans.txt")
    files = glob.glob(pattern)
    data = {}
    for f in files:
        with open(f) as fid:
            lines = (l.strip().split() for l in fid)
            lines = ((l[0], " ".join(l[1:])) for l in lines)
            data.update(lines)
    return data


def path_from_key(key, prefix, ext):
    dirs = key.split("-")
    dirs[-1] = key
    prefix = prefix.replace(args.root_directory, "")
    path = os.path.join(prefix, *dirs)
    return path + os.path.extsep + ext


def clean_text(text):
    return text.strip().lower()


def build_json(path):
    transcripts = load_transcripts(path)
    dirname = os.path.dirname(path)
    basename = os.path.join(
        args.output_directory, os.path.basename(path) + os.path.extsep + "yaml"
    )

    with open(basename, "w") as fid:
        data = []
        for k, t in tqdm.tqdm(transcripts.items()):

            wave_file = path_from_key(k, path, ext="flac")
            t = clean_text(t)
            datum = {"text": t, "audio": wave_file}
            data.append(datum)

        metadata = {"audio_root": args.root_directory}
        output = {"meta": metadata, "data": data}
        yaml.dump(
            output,
            fid,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
            width=float("inf"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess librispeech dataset.")

    parser.add_argument(
        "root_directory",
        default="/mnt/hdd/data/librispeech/",
        help="The dataset is saved in <root_directory>/LibriSpeech.",
    )

    parser.add_argument(
        "--output_directory",
        default=f"{os.path.dirname(os.path.abspath(__file__))}/../data",
        help="The dataset is saved in <output_directory>/LibriSpeech.",
    )

    args = parser.parse_args()

    path = os.path.join(args.root_directory, "LibriSpeech")

    for dataset, dirs in SETS.items():

        print(dataset, dirs)
        for d in dirs:
            print("Preprocessing {}".format(d))
            prefix = os.path.join(path, d)
            build_json(prefix)
