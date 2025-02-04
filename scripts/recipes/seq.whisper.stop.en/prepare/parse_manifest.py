import json
import yaml
from glob import glob
import re

ROOT = "/mnt/hdd/data/stop/manifests/"
MANIFESTS = glob(ROOT + "*.tsv")

for MANIFEST in MANIFESTS:
    data = []
    with open(MANIFEST, "r") as f:
        for line in f:
            # skip header
            if line.startswith("file_id"):
                continue
            line = line.strip()
            if not line:
                continue
            line = line.split("\t")
            path = line[0]
            text = line[4]  #
            # text = line[6] # normalized

            domain = line[1]
            # replace test_0/alarm_test_0/00004403.wav	 to test_0/alarm_test_/00004403.wav
            middle_path = path.split("/")[1]
            middle_path = middle_path.replace("_0", "")

            new_path = path.replace(path.split("/")[1], middle_path)

            sample = {"audio": new_path, "text": text, "domain": domain}
            data.append(sample)

    # save it as yaml
    filename = MANIFEST.split("/")[-1].replace(".tsv", ".yaml")
    with open("scripts/recipes/seq.whisper.stop.en/data/" + filename, "w") as f:
        yaml.dump(
            {
                "data": data,
                "meta": {
                    "len": len(data),
                    "root": "/mnt/hdd/data/stop/manifests/",
                    "source": "https://dl.fbaipublicfiles.com/stop/stop.tar.gz",
                },
            },
            f,
        )
