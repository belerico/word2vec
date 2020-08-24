import mmap
import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .vocab import Vocab


class Word2vecDataset(Dataset):
    def __init__(
        self,
        data: Vocab,
        sentences_path: str,
        sg=1,
        window_size=5,
        shrink_window_size=True,
        ns_size=5,
        mikolov_context=False,
    ):

        self.data = data
        self.sg = sg
        self.window_size = window_size
        self.shrink_window_size = shrink_window_size
        self.ns_size = ns_size
        self.sentences_path = open(sentences_path, "rb")
        self.sentences = np.load(sentences_path, mmap_mode="r")
        # self.mm = mmap.mmap(
        #     self.sentences_path.fileno(), 0, access=mmap.ACCESS_READ
        # )
        self.mikolov_context = mikolov_context

    def __len__(self):
        return self.data.sentence_cnt

    def __getitem__(self, idx):

        wids = self.sentences["arr_" + str(idx)]
        subsampled_wids = []
        for wid in wids:
            if self.data.discard_table[wid] >= np.random.rand():
                subsampled_wids.append(wid)

        self.data.word_cnt -= len(wids) - len(subsampled_wids)

        if subsampled_wids:
            # Shrink window by b
            b = self.window_size
            if self.shrink_window_size:
                b = np.random.randint(1, self.window_size + 1)

            examples = []
            if self.sg:
                if self.mikolov_context:
                    examples = [
                        (
                            target,
                            context,
                            self.data.get_negative_samples(self.ns_size),
                        )
                        for i, target in enumerate(subsampled_wids)
                        for context in subsampled_wids[max(0, i - b) : i]
                        + subsampled_wids[i + 1 : i + b + 1]
                    ]
                else:
                    examples = [
                        (
                            target,
                            context,
                            self.data.get_negative_samples(self.ns_size),
                        )
                        for i, target in enumerate(subsampled_wids)
                        for context in subsampled_wids[max(0, i - b) : i]
                        + subsampled_wids[i + 1 : i + b + 1]
                        if target != context
                        and context in wids[max(i - b, 0) : i + b + 1]
                    ]
            else:
                examples = []
                for i, target in enumerate(subsampled_wids):
                    if self.mikolov_context:
                        context = [
                            c
                            for c in subsampled_wids[max(0, i - b) : i]
                            + subsampled_wids[i + 1 : i + b + 1]
                        ]
                    else:
                        context = [
                            c
                            for c in subsampled_wids[max(0, i - b) : i]
                            + subsampled_wids[i + 1 : i + b + 1]
                            if c
                            in wids[max(0, i - b) : i] + wids[i + 1 : i + b + 1]
                        ]
                    if len(context) > 0:
                        examples.append(
                            (
                                target,
                                context,
                                self.data.get_negative_samples(self.ns_size),
                            )
                        )
            return examples
        else:
            return []

    @staticmethod
    def collate_sg(batches):
        all_target = [t for b in batches for t, _, _ in b]
        all_context = [c for b in batches for _, c, _ in b]
        all_neg = [neg for b in batches for _, _, neg in b]
        # all_lengths = sum([b[1] for b in batches])

        return (
            torch.LongTensor(all_target),
            torch.LongTensor(all_context),
            torch.LongTensor(all_neg),
            # all_lengths,
        )

    @staticmethod
    def collate_cw(batches):
        # all_lengths = sum([b[1] for b in batches])
        if batches[0]:
            all_target = [t for b in batches for t, _, _ in b]
            all_context = [
                torch.LongTensor(c) for b in batches for _, c, _ in b
            ]
            all_neg = [neg for b in batches for _, _, neg in b]

            return (
                torch.LongTensor(all_target),
                torch.LongTensor(pad_sequence(all_context, batch_first=True)),
                torch.LongTensor(all_neg),
                # all_lengths,
            )
        else:
            return ([], [], [])
