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
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    print("predicted_ids")
    print(predicted_ids.size())
    print(predicted_ids)
    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    print(transcription)

    print("@@@@@" * 10)

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

    print(train_dataloader)
    print(eval_dataloaders)

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

    print(eval_dataloaders)

    group_dataloaders = eval_dataloaders

    first_sample = eval_dataloaders["librispeech-dev-clean"].dataset[0]
    print(first_sample)
    print(first_sample["input_features"].shape)
    print(type(first_sample["input_features"]))
    print(input_features.size())


    # covnert first_sample["input_features"] int tensor
    torch_input_features = torch.tensor(first_sample["input_features"]).unsqueeze(0)

    print(torch_input_features.size())
    # compare values of first_sample["input_features"] and input_features
    # input_features_numpy = input_features.squeeze().cpu().numpy()
    print(torch.allclose(torch_input_features, input_features))

    # Check Collate function
    for batch in eval_dataloaders["librispeech-dev-clean"]:
        print(batch)
        print(batch["input_features"][0].size())
        cf_input_features = batch["input_features"].cpu()
        break

    
    # compare values of cf_input_features and input_features
    print(torch.allclose(cf_input_features, input_features))

    with torch.no_grad():
        predicted_ids = model.generate(cf_input_features.to(accelerator.device))

    print("predicted_ids")
    print(predicted_ids.size())
    print(predicted_ids)
    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    print(transcription)

    print("@@@@@" * 10)

    print("🪼🪼🪼🪼🪼🪼🪼🪼🪼🪼🪼🪼")

    for inter_step, batch in enumerate(eval_dataloaders["librispeech-dev-clean"]):
        print("batch")
        print(batch)

        
        
        ref_text = processor.batch_decode(batch["labels"], skip_special_tokens=True)
        print("ref_text")
        print(ref_text)

        print(batch["input_features"].size())

        batch = batch.to(accelerator.device)
        with torch.no_grad():
            predictions = model.generate(**batch)
        
        
        print("out@@@@@@@@@@@@@")
        print(predictions.size())
        print(predictions)
        # predictions[predictions == -100] = (
        #                     processor.tokenizer.pad_token_id
        #                 )
        transcription = processor.batch_decode(predictions, skip_special_tokens=True)

        print(transcription)


        print("#########")
        print(batch["input_features"][0].unsqueeze(0).size())

        with torch.no_grad():
            predictions = model.generate(batch["input_features"][0].unsqueeze(0), max_length=512)
        print(predictions)
        print(predictions.size())


\

        print("🍓🍓🍓🍓🍓🍓🍓🍓🍓🍓🍓🍓🍓")
        input_features = input_features.to(accelerator.device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        print("predicted_ids")
        print(predicted_ids.size())
        print(predicted_ids)
        # Decode token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        print(transcription)

        print("@@@@@" * 10)

        print(batch["input_features"][0].unsqueeze(0))
        print(batch["input_features"][0].unsqueeze(0).size())
        
        print(input_features)
        print(input_features.size())

        # compare values of batch["input_features"][0].unsqueeze(0) and input_features
        print(torch.allclose(batch["input_features"][0].unsqueeze(0), input_features))

        # get difference between batch["input_features"][0].unsqueeze(0) and input_features
        print(torch.sum(torch.abs(batch["input_features"][0].unsqueeze(0) - input_features)))





if __name__ == "__main__":
    main()
