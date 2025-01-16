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
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from accelerate.utils.tqdm import tqdm
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_pt_utils import get_model_param_count

from src.dataset.dataset import AbstractAudioDataset, DataCollatorCTCWithPadding

# disable UserWarning
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def import_module(module_path):
    module_name = module_path.split("/")[-1].split(".")[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: Dict) -> None:

    accelerator = Accelerator(
        log_with="mlflow",
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        project_dir=cfg.train.output_dir,
    )
    accelerator.print(OmegaConf.to_yaml(cfg, resolve=True))

    # Set seed
    set_seed(cfg.train.seed)

    # Load Model
    processor = AutoProcessor.from_pretrained(cfg.model.processor_name_or_path)
    config = AutoConfig.from_pretrained(cfg.model.model_name_or_path)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        cfg.model.model_name_or_path, config=config, ignore_mismatched_sizes=True
    )

    # if cfg.model.freeze_feature_encoder:
    # model.freeze_feature_encoder()

    if cfg.model.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if cfg.model.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    # Load Dataset
    dataset_module = import_module(cfg.data.data_module)

    train_dataloader, eval_dataloaders, train_dataset, eval_datasets = (
        dataset_module.prepare(cfg)
    )

    eval_metrics = {metric: evaluate.load(metric) for metric in cfg.data.eval_metrics}

    optimizer = AdamW(params=model.parameters(), lr=cfg.train.learning_rate)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.train.warmup_steps,
        num_training_steps=(len(train_dataloader) * cfg.train.num_epochs)
        // cfg.train.gradient_accumulation_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    # Each dataloader is prepared separately
    for group, g_dataloader in eval_dataloaders.items():
        eval_dataloaders[group] = accelerator.prepare(g_dataloader)

    group_dataloaders = eval_dataloaders

    print("🪼🪼🪼🪼🪼🪼🪼🪼🪼🪼🪼🪼")
    for inter_step, batch in enumerate(eval_dataloaders["librispeech-dev-clean"]):

        ref_text = processor.batch_decode(batch["labels"], skip_special_tokens=True)
        print("ref_text")

        print(ref_text)

        with torch.no_grad():
            predictions = model.generate(**batch)

        transcription = processor.batch_decode(predictions, skip_special_tokens=True)
        print("transcription")
        print(transcription)


if __name__ == "__main__":
    main()
