from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
from torch.utils.data import Dataset

from .vocab import Vocab


class Word2vecDataset(Dataset):
    def __init__(
        self,
        data: Vocab,
        sg=1,
        window_size=5,
        shrink_window_size=True,
        ns_size=5,
        max_sentence_length=1000,
    ):

        self.eof = False
        self.data = data
        if data.max_sentence_length != max_sentence_length:
            raise Exception(
                "Max sentence lengths differs between Vocab ("
                + str(data.max_sentence_length)
                + ") and parameter ("
                + str(max_sentence_length)
                + ")"
            )
        self.sg = sg
        self.window_size = window_size
        self.shrink_window_size = shrink_window_size
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
            while not self.eof and char_read < self.max_sentence_length and not new_line:
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

            if len(line) > 1:
                words = line.split()
                if len(words) > 1:
                    wids = []
                    subsampled_wids = []
                    for w in words:
                        if w in self.data.word2id:
                            wids.append(self.data.word2id[w])
                            if np.random.rand() < self.data.discard_table[wids[-1]]:
                                subsampled_wids.append(wids[-1])

                    # wids = [
                    #     self.data.word2id[w]
                    #     for w in words
                    #     if w in self.data.word2id
                    # ]
                    # subsampled_wids = [
                    #     wid
                    #     for wid in wids
                    #     if np.random.rand() < self.data.discard_table[wid]
                    # ]

                    # Shrink window by b
                    b = self.window_size
                    if self.shrink_window_size:
                        b = np.random.randint(1, self.window_size + 1)

                    if self.sg:
                        return [
                            (
                                target,
                                context,
                                self.data.get_negative_samples(self.ns_size),
                                len(wids),
                            )
                            for i, target in enumerate(subsampled_wids)
                            for context in subsampled_wids[max(i - b, 0) : i + b + 1]
                            if target != context
                            and context in wids[max(i - b, 0) : i + b + 1]
                        ]
                    else:
                        examples = []
                        for i, target in enumerate(subsampled_wids):
                            context = [
                                c
                                for c in subsampled_wids[max(0, i - b) : i]
                                + subsampled_wids[i + 1 : i + b + 1]
                                if c in wids[max(0, i - b) : i] + wids[i + 1 : i + b + 1]
                            ]
                            if len(context) > 0:
                                examples.append(
                                    (
                                        target,
                                        context,
                                        self.data.get_negative_samples(self.ns_size),
                                        len(wids),
                                    )
                                )
                        return examples

    @staticmethod
    def collate_sg(batches):
        all_target = [t for b in batches for t, _, _, _ in b if len(b) > 0]
        all_context = [c for b in batches for _, c, _, _ in b if len(b) > 0]
        all_neg = [neg for b in batches for _, _, neg, _ in b if len(b) > 0]
        all_lengths = [l for b in batches for _, _, _, l in b if len(b) > 0]

        return (
            torch.LongTensor(all_target),
            torch.LongTensor(all_context),
            torch.LongTensor(all_neg),
            all_lengths,
        )

    @staticmethod
    def collate_cw(batches):
        all_target = [t for b in batches for t, _, _, _ in b if len(b) > 0]
        all_context = [
            torch.LongTensor(c) for b in batches for _, c, _, _ in b if len(b) > 0
        ]
        all_neg = [neg for b in batches for _, _, neg, _ in b if len(b) > 0]
        all_lengths = [l for b in batches for _, _, _, l in b if len(b) > 0]

        if all_context:
            return (
                torch.LongTensor(all_target),
                torch.LongTensor(pad_sequence(all_context, batch_first=True)),
                torch.LongTensor(all_neg),
                all_lengths,
            )
        else:
            return []
