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
    AutoModelForCTC,
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

    config.update(
        {
            "feat_proj_dropout": cfg.model.feat_proj_dropout,
            "attention_dropout": cfg.model.attention_dropout,
            "hidden_dropout": cfg.model.hidden_dropout,
            "final_dropout": cfg.model.final_dropout,
            "mask_time_prob": cfg.model.mask_time_prob,
            "mask_time_length": cfg.model.mask_time_length,
            "mask_feature_prob": cfg.model.mask_feature_prob,
            "mask_feature_length": cfg.model.mask_feature_length,
            "layerdrop": cfg.model.layerdrop,
            "ctc_loss_reduction": cfg.model.ctc_loss_reduction,
            "ctc_zero_infinity": cfg.model.ctc_zero_infinity,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "vocab_size": len(processor.tokenizer),
            "activation_dropout": cfg.model.activation_dropout,
            "add_adapter": cfg.model.add_adapter,
            "ctc_zero_infinity": True,  # should be always True
        }
    )
    model = AutoModelForCTC.from_pretrained(
        cfg.model.model_name_or_path, config=config, ignore_mismatched_sizes=True
    )

    if cfg.model.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # Load Dataset
    dataset_module = import_module(cfg.data.data_module)

    train_dataloader, eval_dataloaders, train_dataset, eval_datasets = (
        dataset_module.prepare(cfg)
    )

    eval_metrics = {metric: evaluate.load(metric) for metric in cfg.data.eval_metrics}

    if accelerator.is_main_process:

        # must be called before accelerator.get_tracker
        accelerator.init_trackers(
            project_name=(
                cfg.train.project_name if cfg.train.project_name else "default"
            ),
            init_kwargs={
                "mlflow": {
                    "run_name": cfg.train.run_name if cfg.train.run_name else None
                }
            },
        )
        mlflow_tracker = accelerator.get_tracker("mlflow", unwrap=True)

        run_info = mlflow_tracker.info
        host_url = mlflow.get_tracking_uri()
        experiment_id = run_info.experiment_id
        run_id = run_info.run_id
        run_name = run_info.run_name
        experment_url = f"{host_url}/#/experiments/{experiment_id}"
        run_url = f"{experment_url}/runs/{run_id}"

        accelerator.print(f"🏃 View run {run_name} at: {run_url}.")
        accelerator.print(f"🧪 View experiment at: {experment_url}.")

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
    for group, g_dataloader in eval_dataloaders.items():
        eval_dataloaders[group] = accelerator.prepare(g_dataloader)

    print(eval_dataloaders)


    # group_dataloaders = {
    #     "librispeech-test-clean": test_clean_dataloader,
    #     "librispeech-dev-clean": dev_clean_dataloader,
    # }
    group_dataloaders = eval_dataloaders

    accelerator.register_for_checkpointing(model, optimizer, lr_scheduler)

    accelerator.print("============== Start Training ==============")

    num_update_steps_per_epoch = (
        len(train_dataloader) // cfg.train.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_total_update = num_update_steps_per_epoch * cfg.train.num_epochs

    total_batch_size = (
        cfg.train.gradient_accumulation_steps
        * cfg.train.per_device_train_batch_size
        * accelerator.num_processes
    )

    accelerator.print(f"Number of training examples: {len(train_dataset)}")
    accelerator.print(f"Number of evaluation examples: ")
    for group, g_dataset in eval_datasets.items():
        accelerator.print(f"- {group}: {len(g_dataset)}")
    accelerator.print(f"Number of training epochs: {cfg.train.num_epochs}")
    accelerator.print(f"Batch size per device: {cfg.train.per_device_train_batch_size}")
    accelerator.print(
        f"Gradient Accumulation steps: {cfg.train.gradient_accumulation_steps}"
    )
    accelerator.print(
        f"Total batch size: (batch_size ({cfg.train.per_device_train_batch_size}) * "
        f"accumulate_step ({cfg.train.gradient_accumulation_steps}) * "
        f"world_size ({accelerator.num_processes})) = {total_batch_size}"
    )

    accelerator.print(
        f"Update steps per epoch: Training sample ({len(train_dataset)}) // Total batch size ({total_batch_size}) = {num_update_steps_per_epoch}"
    )

    accelerator.print(
        f"Total number of updates: Upate steps per epoch ({num_update_steps_per_epoch}) * Training Epoch ({cfg.train.num_epochs}) = {num_total_update}"
    )

    accelerator.print(
        f"Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
    )

    accelerator.print(
        f"Model: {cfg.model.model_name_or_path}, Processor: {cfg.model.processor_name_or_path}"
    )

    accelerator.print(
        "============================================"
    )  # mlflow._log_url()

    global_step = 0
    eval_flag = True
    for epoch in range(cfg.train.num_epochs):
        model.train()
        accumulated_loss = 0.0
        epoch_loss = 0.0

        train_dataloader_with_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{cfg.train.num_epochs}",
            leave=True,
        )

        for step, batch in enumerate(train_dataloader_with_bar):
            # We could avoid this line since we set the accelerator with `device_placement=True`.

            batch.to(accelerator.device)
            with accelerator.accumulate(model):

                outputs = model(**batch)
                loss = outputs.loss
                # for gradient accumulation
                # avg_loss += loss.item()
                loss_tensor = torch.tensor(loss.item()).to(accelerator.device)
                accumulated_loss += loss_tensor

                epoch_loss += loss.item()

                accelerator.backward(loss)

                # clip gradient (before optimizer.step)
                # https://github.com/huggingface/accelerate/issues/641
                if accelerator.sync_gradients:
                    if (
                        cfg.train.do_grad_norm_clip
                        and cfg.train.max_grad_norm is not None
                    ):
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), cfg.train.max_grad_norm
                        )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Acculumate loss for logging
            # https://github.com/huggingface/accelerate/issues/639
            if accelerator.sync_gradients:
                global_step += 1
                # gather the loss from all replicas
                accumulated_loss = accelerator.gather(
                    accumulated_loss / accelerator.gradient_accumulation_steps
                )
                accumulated_loss = accumulated_loss.mean().item()

                train_dataloader_with_bar.set_postfix_str(
                    f"loss: {accumulated_loss:.4f}, global_step: {global_step}, grad_norm: {grad_norm:.4f}"
                )

                if cfg.train.log_with:
                    accelerator.log(
                        {
                            "loss_train": accumulated_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "grad_norm": grad_norm.item(),
                            "epoch": epoch
                            + ((step + 1) / len(train_dataloader_with_bar)),
                        },
                        step=global_step,
                    )
                accumulated_loss = 0.0
                eval_flag = True

            if ((global_step + 1) % cfg.train.eval_steps == 0) and eval_flag:

                for group, g_eval_dataloader in group_dataloaders.items():

                    eval_dataloader_with_bar = tqdm(
                        g_eval_dataloader,
                        desc=f"Evaluation on {group} @ step {global_step+1}",
                        leave=True,
                    )

                    model.eval()
                    prediction_strs = []
                    label_strs = []
                    for inter_step, batch in enumerate(eval_dataloader_with_bar):
                        # We could avoid this line since we set the accelerator with `device_placement=True`.
                        batch.to(accelerator.device)
                        with torch.no_grad():
                            outputs = model(**batch)

                        predictions = outputs.logits.argmax(dim=-1)
                        predictions[predictions == -100] = (
                            processor.tokenizer.pad_token_id
                        )
                        predictions = processor.batch_decode(predictions)

                        labels = batch["labels"]
                        labels[labels == -100] = processor.tokenizer.pad_token_id
                        labels = processor.batch_decode(labels,  group_tokens=False)

                        predictions = accelerator.gather_for_metrics(predictions)
                        labels = accelerator.gather_for_metrics(labels)

                        prediction_strs.extend(predictions)
                        label_strs.extend(labels)

                    if accelerator.is_main_process:

                        metrics = {
                            k: v.compute(
                                predictions=prediction_strs, references=label_strs
                            )
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
                        with open(
                            save_dir / f"{global_step+1}_{group}.txt", mode="w"
                        ) as f:
                            for p, r in zip(prediction_strs, label_strs):
                                f.write(
                                    json.dumps(
                                        {"prediction": p, "reference": r},
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                        accelerator.print(
                            f"prediction: {save_dir}/{global_step+1}_{group}.txt"
                        )
                        mlflow.log_artifact(save_dir / f"{global_step+1}_{group}.txt")

                if accelerator.is_main_process:
                    # accelerator.save_state(save_dir / f"checkpoint-{global_step+1}")
                    accelerator.print(
                        "Saving checkpoint...",
                        str(save_dir / f"checkpoint-{global_step+1}"),
                    )
                    accelerator.unwrap_model(model).save_pretrained(
                        save_dir / f"checkpoint-{global_step+1}"
                    )
                    processor.save_pretrained(save_dir / f"checkpoint-{global_step+1}")

                    if cfg.train.upload_checkpoint:
                        accelerator.print("Uploading checkpoint to mlflow...")
                        try:
                            mlflow.log_artifact(
                                save_dir / f"checkpoint-{global_step+1}"
                            )
                        except Exception as e:
                            accelerator.print(
                                f"Failed to upload checkpoint to mlflow: {e}"
                            )
                        accelerator.print("[Done]")

                    # remove old checkpoint
                    if cfg.train.checkpoint_save_limit:
                        for checkpoint in save_dir.glob("checkpoint-*"):

                            # remove old checkpoint to keep checkpoint_save_limit
                            checkpoints = list(save_dir.glob("checkpoint-*"))
                            checkpoints.sort(key=lambda x: int(x.stem.split("-")[1]))
                            if len(checkpoints) > cfg.train.checkpoint_save_limit:
                                for checkpoint in checkpoints[
                                    : -cfg.train.checkpoint_save_limit
                                ]:
                                    accelerator.print(
                                        f"Remove old checkpoint: {checkpoint}"
                                    )
                                    shutil.rmtree(checkpoint)

                # Prevent multiple evaluation per step (e.g. when using gradient accumulation)
                # only evaluate once per global step (not per gradient accumulation step)
                eval_flag = False
                model.train()

        accelerator.print(
            f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_dataloader)}"
        )

        if cfg.train.log_with:
            accelerator.log(
                {
                    "epoch_loss_train": epoch_loss / len(train_dataloader),
                    "epoch": epoch,
                },
                step=epoch,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
