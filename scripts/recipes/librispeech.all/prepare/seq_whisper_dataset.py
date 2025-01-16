import torchaudio
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoProcessor
from dataclasses import dataclass, field
from typing import Dict, List, Union
from src.dataset.dataset import (
    AbstractAudioDataset,
)
import torch



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: AutoProcessor
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [
            {model_input_name: feature[model_input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor(
                [feature["attention_mask"] for feature in features]
            )

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



class LibriSpeechSeq2SeqDataset(AbstractAudioDataset):

    def __init__(
        self,
        list_of_pairs,
        root_dir,
        processor,
        max_input_length_in_sec=20,
        max_sample=99999999,
        forward_attention_mask=True,
    ):
        super().__init__(
            list_of_pairs, root_dir, max_sample
        )
        self.processor = processor
        self.max_input_length_in_sec = max_input_length_in_sec
        self.max_sample = max_sample
        self.forward_attention_mask = forward_attention_mask

    def transform(self, datum):

        speech, sample_rate = torchaudio.load(str(datum["audio"]).strip())
        if sample_rate != 16000:
            speech = torchaudio.transforms.Resample(sample_rate, 16000).forward(speech)
        speech = speech[0]

        inputs = self.processor.feature_extractor(
            speech,
            sampling_rate=16000,
            truncate=True,
            return_attention_mask=self.forward_attention_mask,
        )

        label = self.processor.tokenizer(datum["text"]).input_ids

        sample = {
            "input_features": inputs["input_features"][0],
            "labels": label,
        }

        if self.forward_attention_mask:
            sample["attention_mask"] = inputs["attention_mask"][0]

        return sample


def prepare(cfg, processor):

    forward_attention_mask = (
        getattr(cfg.model, "apply_spec_augment", False)
        and getattr(cfg.model, "mask_time_prob", 0) > 0
    )
    # FIXME
    forward_attention_mask = True

    collate_fn = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=cfg.model.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )

    # For trainset, we concatenate multiple datasets
    train_datasets = []
    for train_yaml in cfg.data.train_datasets:
        with open(train_yaml) as f:
            train_list_of_pairs = yaml.load(f, Loader=yaml.FullLoader)

        train_datasets.append(
            LibriSpeechSeq2SeqDataset(
                train_list_of_pairs["data"], cfg.data.audio_root_path, processor, forward_attention_mask
            )
        )

    train_dataset = ConcatDataset(train_datasets)

    print(f"train_dataset: {len(train_dataset)}")

    # For evalset, we use separate datasets

    eval_datasets = {
        "librispeech-test-clean": [],
        "librispeech-dev-clean": [],
    }

    # print(cfg.data.eval_datasets["test-clean"])
    print(cfg.data.eval_datasets)

    with open(cfg.data.eval_datasets["test-clean"]) as f:
        test_clean_list_of_pairs = yaml.load(f, Loader=yaml.FullLoader)

    eval_datasets["librispeech-test-clean"] = LibriSpeechSeq2SeqDataset(
        test_clean_list_of_pairs["data"], cfg.data.audio_root_path, processor, forward_attention_mask
    )

    with open(cfg.data.eval_datasets["dev-clean"]) as f:
        dev_clean_list_of_pairs = yaml.load(f, Loader=yaml.FullLoader)

    eval_datasets["librispeech-dev-clean"] = LibriSpeechSeq2SeqDataset(
        dev_clean_list_of_pairs["data"], cfg.data.audio_root_path, processor, forward_attention_mask
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.train.per_device_train_batch_size,
        drop_last=True,
    )

    test_clean_dataloader = DataLoader(
        eval_datasets["librispeech-test-clean"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=cfg.train.per_device_eval_batch_size,
        drop_last=False,
    )
    dev_clean_dataloader = DataLoader(
        eval_datasets["librispeech-dev-clean"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=cfg.train.per_device_eval_batch_size,
        drop_last=False,
    )

    eval_dataloaders = {
        "librispeech-test-clean": test_clean_dataloader,
        "librispeech-dev-clean": dev_clean_dataloader,
    }
    return train_dataloader, eval_dataloaders, train_dataset, eval_datasets
