# import mmap
import pickle
import random

import numpy as np
import torch

# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, Dataset

from .vocab import Vocab


class Word2vecDataset(Dataset):
    def __init__(
        self,
        data: Vocab,
        # sentences_path: str,
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
        self.sentences = pickle.load(open("./sentences/sentences.pkl", "rb"))
        self.mikolov_context = mikolov_context

    def __len__(self):
        return self.data.sentence_cnt

    def __getitem__(self, idx):
        subsampled_wids, len_wids = self.sentences[idx]
        # Shrink window by b
        b = self.window_size
        if self.shrink_window_size:
            b = np.random.randint(1, self.window_size + 1)

        examples = []
        if self.sg:
            examples = [
                (target, context, self.data.get_negative_samples(self.ns_size),)
                for i, target in enumerate(subsampled_wids)
                for context in subsampled_wids[max(0, i - b) : i]
                + subsampled_wids[i + 1 : i + b + 1]
            ]
        else:
            examples = []
            for i, target in enumerate(subsampled_wids):
                context = (
                    subsampled_wids[max(0, i - b) : i]
                    + subsampled_wids[i + 1 : i + b + 1]
                )
                if context:
                    examples.append(
                        (
                            target,
                            context + [0 for _ in range(2 * b - len(context))],
                            self.data.get_negative_samples(self.ns_size),
                        )
                    )
        return examples, len_wids

    def collate(self, batches):
        return (
            torch.LongTensor([t for b in batches for t, _, _ in b[0]]),
            torch.LongTensor([c for b in batches for _, c, _ in b[0]]),
            torch.LongTensor([neg for b in batches for _, _, neg in b[0]]),
            sum([b[1] for b in batches]),
        )

    # @staticmethod
    # def collate_cw(batches):
    #     # all_lengths = sum([b[1] for b in batches])
    #     all_target = [t for b in batches for t, _, _ in b]
    #     all_context = [c for b in batches for _, c, _ in b]
    #     all_neg = [neg for b in batches for _, _, neg in b]

    #     return (
    #         torch.LongTensor(all_target),
    #         torch.LongTensor(all_context),
    #         torch.LongTensor(all_neg),
    #         # all_lengths,
    #     )
