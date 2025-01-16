import torchaudio
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoProcessor

from src.dataset.dataset import AbstractAudioDataset, DataCollatorCTCWithPadding


class LibriSpeechDataset(AbstractAudioDataset):

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
            speech = torchaudio.transforms.Resample(sample_rate, 16000).forward(speech)
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


def prepare(cfg):

    processor = AutoProcessor.from_pretrained(cfg.model.processor_name_or_path)

    collate_fn = DataCollatorCTCWithPadding(processor=processor)

    # For trainset, we concatenate multiple datasets
    train_datasets = []
    for train_yaml in cfg.data.train_datasets:
        with open(train_yaml) as f:
            train_list_of_pairs = yaml.load(f, Loader=yaml.FullLoader)

        train_datasets.append(
            LibriSpeechDataset(
                train_list_of_pairs["data"], cfg.data.audio_root_path, processor
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

    eval_datasets["librispeech-test-clean"] = LibriSpeechDataset(
        test_clean_list_of_pairs["data"], cfg.data.audio_root_path, processor
    )

    with open(cfg.data.eval_datasets["dev-clean"]) as f:
        dev_clean_list_of_pairs = yaml.load(f, Loader=yaml.FullLoader)

    eval_datasets["librispeech-dev-clean"] = LibriSpeechDataset(
        dev_clean_list_of_pairs["data"], cfg.data.audio_root_path, processor
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
