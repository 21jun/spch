import argparse
import datetime
import functools
import importlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import accelerate
import evaluate
import hydra
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import transformers
import yaml
from abs.dataset import AbstractAudioDataset, DataCollatorCTCWithPadding
from accelerate import Accelerator, DistributedType, PartialState
from accelerate.utils import gather_object, set_seed
from accelerate.utils.tqdm import tqdm
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import (AutoConfig, AutoFeatureExtractor, AutoModelForCTC,
                          AutoProcessor, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from transformers.trainer_pt_utils import get_model_param_count

# disable UserWarning
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: Dict) -> None:

    # (OmegaConf.to_yaml(cfg, resolve=True))

    # Set seed
    set_seed(cfg.train.seed)

    # Load Model
    processor = AutoProcessor.from_pretrained(cfg.model.processor_name_or_path)
    config = AutoConfig.from_pretrained(cfg.model.model_name_or_path)
    model = AutoModelForCTC.from_pretrained(
        cfg.model.model_name_or_path, config=config, ignore_mismatched_sizes=True
    )

    if cfg.model.freeze_feature_encoder:
        model.freeze_feature_encoder()

    class MyDataset(AbstractAudioDataset):

        def __init__(
            self,
            list_of_pairs,
            root_dir,
            processor,
            max_input_length_in_sec=20,
            max_sample=99999999,
        ):
            super().__init__(list_of_pairs, root_dir, max_sample)
            self.processor = processor
            self.max_input_length_in_sec = max_input_length_in_sec
            self.max_sample = max_sample

        def transform(self, datum):

            speech, sample_rate = torchaudio.load(str(datum["audio"]).strip())
            if sample_rate != 16000:
                speech = torchaudio.transforms.Resample(sample_rate, 16000).forward(
                    speech
                )
            speech = speech[0]
            speech = speech[: 16000 * self.max_input_length_in_sec]

            input_value = self.processor.feature_extractor(
                speech,
                sampling_rate=16000,
                truncate=True,
                max_length=self.max_input_length_in_sec * 16000,
            )

            # upper case
            label = self.processor(text=datum["text"].upper()).input_ids

            sample = {
                "input_values": input_value["input_values"][0],
                "labels": label,
            }

            return sample

    # Load Dataset
    with open(cfg.data.eval_data_paths[0]) as f:
        eval_set_pairs = yaml.load(f, Loader=yaml.FullLoader)

    evalset = MyDataset(
        eval_set_pairs["data"], cfg.data.audio_root_path, processor, max_sample=999999
    )
    print(f"Number of evaluation examples: {len(evalset)}")
    print(evalset[0])

    collate_fn = DataCollatorCTCWithPadding(processor=processor)

    eval_dataloader = DataLoader(
        evalset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=cfg.train.per_device_eval_batch_size,
        drop_last=False,
    )

    eval_metrics = {metric: evaluate.load(metric) for metric in cfg.data.eval_metrics}

    accelerator = Accelerator()

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.eval()

    eval_dataloader_with_bar = tqdm(
        eval_dataloader,
        desc=f"Evaluation",
        leave=True,
    )

    prediction_strs = []
    label_strs = []

    with torch.no_grad():
        for batch in eval_dataloader_with_bar:
            outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            predictions[predictions == -100] = processor.tokenizer.pad_token_id
            predictions = processor.batch_decode(predictions)
            predictions = accelerator.gather_for_metrics(predictions)
            prediction_strs.extend(predictions)

    # Post process
    prediction_strs = [pred.strip() for pred in prediction_strs]

    label_strs = [x["text"].upper() for x in eval_set_pairs["data"]]

    label_strs = label_strs[: len(prediction_strs)]

    for p, l in zip(prediction_strs, label_strs):
        print(f"P: {p}")
        print(f"R: {l}")

    if accelerator.is_main_process:

        metrics = {
            k: v.compute(predictions=prediction_strs, references=label_strs)
            for k, v in eval_metrics.items()
        }

        print(metrics)

    exit()

    for inter_step, batch in enumerate(eval_dataloader_with_bar):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        predictions[predictions == -100] = processor.tokenizer.pad_token_id
        predictions = processor.batch_decode(predictions)

        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        labels = processor.batch_decode(labels)

        predictions = accelerator.gather_for_metrics(predictions)
        labels = accelerator.gather_for_metrics(labels)

        prediction_strs.extend(predictions)
        label_strs.extend(labels)

    if accelerator.is_main_process:

        metrics = {
            k: v.compute(predictions=prediction_strs, references=label_strs)
            for k, v in eval_metrics.items()
        }
        accelerator.print(
            global_step + 1,
            metrics,
            group,
        )
        # add group prefix to metrics key
        metrics = {f"{group}_{k}": v for k, v in metrics.items()}

        accelerator.log(metrics, step=global_step + 1)

        save_dir = Path(cfg.train.output_dir)
        # dump into mlflow artifact
        with open(save_dir / f"{global_step+1}_{group}.txt", mode="w") as f:
            for p, r in zip(prediction_strs, label_strs):
                f.write(
                    json.dumps(
                        {"prediction": p, "reference": r},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        accelerator.print(f"prediction: {save_dir}/{global_step+1}_{group}.txt")
        mlflow.log_artifact(save_dir / f"{global_step+1}_{group}.txt")


if __name__ == "__main__":
    main()
