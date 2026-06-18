import json

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self._offsets = self._build_offsets(data_path)

    @staticmethod
    def _build_offsets(data_path):
        offsets = []
        with open(data_path, "rb") as f:
            offsets.append(0)
            while f.readline():
                offsets.append(f.tell())
        return offsets

    def __len__(self):
        return len(self._offsets) - 1

    def __getitem__(self, index: int):
        with open(self.data_path, "rb") as f:
            f.seek(self._offsets[index])
            line = f.readline().decode("utf-8")

        sample = json.loads(line)
        text = f"{self.tokenizer.bos_token}{sample['text']}{self.tokenizer.eos_token}"
        input_ids = self.tokenizer(text).data["input_ids"][: self.max_length]

        text_len = len(input_ids)
        padding_len = self.max_length - text_len
        input_ids = input_ids + [self.padding] * padding_len
        loss_mask = [1] * text_len + [0] * padding_len
        attention_mask = [1] * text_len + [0] * padding_len

        input_ids = np.array(input_ids)
        x = np.array(input_ids[:-1]).astype(np.int64)
        y = np.array(input_ids[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        attention_mask = np.array(attention_mask[:-1]).astype(np.int64)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(loss_mask),
            torch.from_numpy(attention_mask),
        )
