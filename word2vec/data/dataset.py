# import mmap
import pickle

import torch

# from torch.nn.utils.rnn import pad_sequence
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
    ):

        self.data = data
        self.sg = sg
        self.window_size = window_size
        self.shrink_window_size = shrink_window_size
        self.ns_size = ns_size
        self.sentences = pickle.load(open(sentences_path, "rb"))

    def __len__(self):
        return self.data.sentence_cnt

    def __getitem__(self, idx):
        subsampled_wids, len_wids = self.sentences[idx]
        # Shrink window by b
        b = 0
        if self.shrink_window_size:
            b = (
                -1
                * self.data.rng.integers(
                    low=0,
                    high=self.window_size - 1,
                    size=len(subsampled_wids),
                    endpoint=True,
                )
                + self.window_size
            )

        examples = []
        if self.sg:
            examples = [
                (
                    target,
                    context,
                    self.data.get_negative_samples(target, self.ns_size),
                )
                for i, target in enumerate(subsampled_wids)
                for context in subsampled_wids[max(0, i - b[i]) : i]
                + subsampled_wids[i + 1 : i + b[i] + 1]
            ]
        else:
            for i, target in enumerate(subsampled_wids):
                context = (
                    subsampled_wids[max(0, i - b[i]) : i]
                    + subsampled_wids[i + 1 : i + b[i] + 1]
                )
                examples.append(
                    (
                        target,
                        context
                        + [
                            0
                            for _ in range(2 * self.window_size - len(context))
                        ],
                        self.data.get_negative_samples(target, self.ns_size),
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
