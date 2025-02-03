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
print(f"Torch version: {torch.__version__}")


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

    processor.tokenizer.set_prefix_tokens(language=cfg.data.language, task="transcribe")
    model.generation_config.language = cfg.data.language
    model.generation_config.task = "transcribe"

    # Load Dataset
    dataset_module = import_module(cfg.data.data_module)

    train_dataloader, eval_dataloaders, train_dataset, eval_datasets = (
        dataset_module.prepare(cfg, processor)  # Set processor
    )

    # iter over trian_dataloader to get the first batch
    # for batch in train_dataloader:
    #     print(batch)
    #     break

    # exit()

    eval_metrics = {metric: evaluate.load(metric) for metric in cfg.data.eval_metrics}

    model = accelerator.prepare(model)
    # Each dataloader is prepared separately
    for group, g_dataloader in eval_dataloaders.items():
        eval_dataloaders[group] = accelerator.prepare(g_dataloader)

    group_dataloaders = eval_dataloaders

    accelerator.print("============= Start Generation ==============")

    for group, g_eval_dataloader in group_dataloaders.items():

        eval_dataloader_with_bar = tqdm(
            g_eval_dataloader,
            desc=f"Inferencing on {group}",
            leave=True,
        )

        model.eval()
        prediction_strs = []
        other_hypotheses = []
        label_strs = []
        NUM_BEAMS = 5
        for inter_step, batch in enumerate(eval_dataloader_with_bar):

            with torch.no_grad():
                pred_ids = accelerator.unwrap_model(model).generate(
                    **batch, num_beams=NUM_BEAMS, num_return_sequences=NUM_BEAMS
                )

            # print(pred_ids)

            predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)
            # print(predictions)

            predictions = [p.strip() for p in predictions]
            predictions = [p.lower() for p in predictions]
            # remove special chars
            predictions = [re.sub(r"[^\w\s']", "", p) for p in predictions]

            labels = processor.batch_decode(batch["labels"], skip_special_tokens=True)
            labels = [l.strip() for l in labels]
            labels = [l.lower() for l in labels]
            labels = [re.sub(r"[^\w\s']", "", p) for p in labels]

            predictions = accelerator.gather_for_metrics(predictions)
            labels = accelerator.gather_for_metrics(labels)

            # device predictions into
            beam_predictions= [
                predictions[x:x + NUM_BEAMS] for x in range(0, len(predictions), NUM_BEAMS)
            ]
            best_hyp = [beam[0] for beam in beam_predictions]
            other_hypothesis = [beam[1:] for beam in beam_predictions]
            
            # if accelerator.is_main_process:
            #     if len(best_hyp) != len(labels):

            #         print(f"best_hyp: {len(best_hyp)}")
            #         print(f"labels: {len(labels)}")
            #         print(f"other_hypothesis: {len(other_hypothesis)}")
            #         print(best_hyp)
            #         print("---------------")
            #         print(labels)
            #         print("------------------------------------------")



            prediction_strs.extend(best_hyp)
            other_hypotheses.extend(other_hypothesis)
            label_strs.extend(labels)


        if accelerator.is_main_process:
            
            print(len(prediction_strs))
            print(len(label_strs))
            print(len(other_hypotheses))

            save_dir = Path(cfg.train.output_dir)
            # dump into mlflow artifact
            with open(save_dir / f"_{group}.txt", mode="w") as f:
                for p, r, oh in zip(prediction_strs, label_strs, other_hypotheses):
                    f.write(
                        json.dumps(
                            {"best_hyp": p, "text": r, "other_hyps": oh},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            accelerator.print(f"prediction: {save_dir}/_{group}.txt")


if __name__ == "__main__":
    main()
