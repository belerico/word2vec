import torch
from torch.utils.data import Dataset
from .vocab import Vocab
import numpy as np


class Word2vecDataset(Dataset):
    def __init__(
        self, data: Vocab, window_size=5, ns_size=5, max_sentence_length=1000,
    ):
        self.eof = False
        self.data = data
        self.window_size = window_size
        self.ns_size = ns_size
        self.train_file = open(data.train_file, encoding="utf8")
        self.max_sentence_length = max_sentence_length

    def __len__(self):
        return self.data.sentence_cnt

    def __getitem__(self, idx):
        while True:
            char_read = 0
            new_line = False
            line = ""

            if self.eof:
                self.train_file.seek(0, 0)
                self.eof = False

            # Read file in chunk or until a new line
            while (
                not self.eof
                and char_read < self.max_sentence_length
                and not new_line
            ):
                char = self.train_file.read(1)
                if char == "\n":
                    new_line = True
                elif char == "":
                    self.eof = True
                else:
                    line += char
                    char_read += 1

            if not new_line:
                # If a word is truncated after "max_sentence_length" chars,
                # read until any whitespace is found
                whitespace = False
                while not whitespace:
                    char = self.train_file.read(1)
                    if char.isspace() or not char:
                        whitespace = True
                    else:
                        line += char

            if line != "\n" and len(line) > 1:
                words = line.split()
                if len(words) > 1:
                    wids = [
                        self.data.word2id[w]
                        for w in words
                        if w in self.data.word2id
                        and np.random.rand()
                        < self.data.discard_table[self.data.word2id[w]]
                    ]
                    # Shrink window by b
                    b = np.random.randint(0, self.window_size)
                    return [
                        (
                            target,
                            context,
                            self.data.get_negative_samples(
                                target, context, self.ns_size
                            ),
                        )
                        for i, target in enumerate(wids)
                        for j, context in enumerate(
                            wids[max(i - b, 0) : i + b]
                        )
                        if target != context
                    ]

    @staticmethod
    def collate(batches):
        all_target = [t for b in batches for t, _, _ in b if len(b) > 0]
        all_context = [c for b in batches for _, c, _ in b if len(b) > 0]
        all_neg = [neg for b in batches for _, _, neg in b if len(b) > 0]

        return (
            torch.LongTensor(all_target),
            torch.LongTensor(all_context),
            torch.LongTensor(all_neg),
        )
