import json

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=None):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else 0
        )
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
        text = (
            f"{self.tokenizer.bos_token}"
            f"{sample['text']}"
            f"{self.tokenizer.eos_token}"
        )
        input_ids = self.tokenizer(text,add_special_tokens=False).data["input_ids"][: self.max_length]

        text_len = len(input_ids)
        padding_len = self.max_length - text_len

        input_ids = [self.padding] * padding_len + input_ids
        loss_mask = [0] * padding_len + [1] * text_len
        attention_mask = [0] * padding_len + [1] * text_len

        input_ids = np.array(input_ids)
        x = np.array(input_ids[:-1]).astype(np.int64)
        y = np.array(input_ids[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)*np.array(loss_mask[:-1]).astype(np.int64)
        attention_mask = np.array(attention_mask[:-1]).astype(np.int64)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(loss_mask),
            torch.from_numpy(attention_mask),
        )


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=None):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length or 513
        self.padding = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else 0
        )
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

    def generate_loss_mask(self, input_ids):
        mask = [0] * len(input_ids)
        assistant_sequence = self.tokenizer("<|im_start|>assistant\n",add_special_tokens=False).data["input_ids"]
        assistant_sequence_length = len(assistant_sequence)
        token_count = len(input_ids)
        index = 0

        while index <= token_count - assistant_sequence_length:
            match = True
            for offset in range(assistant_sequence_length):
                if input_ids[index + offset] != assistant_sequence[offset]:
                    match = False
                    break

            if match:
                answer_end = None
                for token_index in range(index + assistant_sequence_length, token_count):
                    if input_ids[token_index] == self.tokenizer.eos_token_id:
                        answer_end = token_index
                        break

                if answer_end is not None:
                    answer_start = index + assistant_sequence_length
                    if answer_start <= answer_end:
                        for mask_index in range(answer_start, answer_end + 1):
                            if mask_index < len(mask):
                                mask[mask_index] = 1
                index += assistant_sequence_length
            else:
                index += 1

        return mask

    def __getitem__(self, index: int):
        with open(self.data_path, "rb") as f:
            f.seek(self._offsets[index])
            line = f.readline().decode("utf-8")

        sample = json.loads(line)
        text = self.tokenizer.apply_chat_template(
            sample,
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = self.tokenizer(text, add_special_tokens=False).data["input_ids"][: self.max_length]

        text_len = len(input_ids)
        padding_len = self.max_length - text_len
        input_ids = [self.padding] * padding_len+input_ids
        attention_mask = [0] * padding_len+[1] * text_len
        loss_mask = self.generate_loss_mask(input_ids)

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
