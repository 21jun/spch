import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
import yaml
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoProcessor, Wav2Vec2Processor

AUDIO_COLUMN_NAME = "audio"
TEXT_COLUMN_NAME = "text"


class AbstractAudioDataset(Dataset):

    def __init__(self, list_of_pairs, root_dir, max_sample=99999999):
        self.list_of_pairs = list_of_pairs
        self.root_dir = root_dir
        self.max_sample = max_sample
        self._load(list_of_pairs, root_dir, max_sample)

    def _load(self, list_of_pairs, root_dir, max_sample):
        self.data = []
        for pair in list_of_pairs:
            self.data.append(
                {
                    "audio": Path(root_dir) / pair[AUDIO_COLUMN_NAME],
                    "text": pair[TEXT_COLUMN_NAME],
                }
            )
        self.data = self.data[:max_sample]

    def __getitem__(self, idx):
        return self.transform(self.data[idx])

    def __len__(self):
        return len(self.data)

    def transform(self, datum):
        return datum


class Wav2Vec2CTCDataset(AbstractAudioDataset):
    def __init__(
        self,
        list_of_pairs,
        root_dir,
        processor,
        max_input_length_in_sec=20,
        max_sample=99999999,
    ):
        super().__init__(list_of_pairs, root_dir)
        self.processor = processor
        self.max_input_length_in_sec = max_input_length_in_sec
        self.max_sample = max_sample

    def transform(self, datum):

        speech, sample_rate = torchaudio.load(str(datum["audio"]).strip())
        if sample_rate != 16000:
            speech = torchaudio.transforms.Resample(sample_rate, 16000).forward(speech)
        speech = speech[0]
        speech = speech[: 16000 * self.max_input_length_in_sec]

        input_value = self.processor.feature_extractor(
            speech,
            sampling_rate=16000,
            truncate=True,
            max_length=self.max_input_length_in_sec * 16000,
        )

        label = self.processor(text=datum["text"]).input_ids

        sample = {
            "input_values": input_value["input_values"][0],
            "labels": label,
        }

        return sample


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    feature_extractor_input_name: Optional[str] = "input_values"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {
                self.feature_extractor_input_name: feature[
                    self.feature_extractor_input_name
                ]
            }
            for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch


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


if __name__ == "__main__":

    # load test-clean.yaml
    with open("test-clean.yaml") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    dataset = AbstractAudioDataset(data["data"], data["meta"]["audio_root"])
    print(dataset[0])

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    dataset = Wav2Vec2CTCDataset(data["data"], data["meta"]["audio_root"], processor)
    print(dataset[0])
