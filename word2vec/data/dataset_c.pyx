# import mmap
import pickle

import fastrand

from torch.utils.data import Dataset

from .vocab import Vocab

from cpython.array cimport array, clone

import cython

fastrand.pcg32_seed(42)

@cython.boundscheck(False)
@cython.wraparound(False)
class Word2vecDataset(Dataset):

    def __init__(
        self,
        data: Vocab,
        str sentences_path,
        bint sg=1,
        int window_size=5,
        bint shrink_window_size=True,
        int ns_size=5,
    ):

        self.data = data
        self.sg = sg
        self.window_size = window_size
        self.shrink_window_size = shrink_window_size
        self.ns_size = ns_size
        self.sentences = pickle.load(open(sentences_path, "rb"))
        self.context = clone(array("l"), 2 * self.window_size + 1, zero=True)

    def __len__(self):
        return self.data.sentence_cnt

    def __getitem__(self, long idx):
        cdef int len_wids
        cdef long a, i, j, target, c_idx
        cdef int[::1] b
        cdef list subsampled_wids, examples = []

        subsampled_wids, len_wids = self.sentences[idx]

        b = clone(array("i"), len(subsampled_wids), zero=False)
        # Shrink window by b
        if self.shrink_window_size:
            for i in range(len(subsampled_wids)):
                b[i] = self.window_size - fastrand.pcg32bounded(self.window_size)
        else:
            for i in range(len(subsampled_wids)):
                b[i] = 0

        if not self.sg:
            for i in range(len(subsampled_wids)):
                for j in range(len(self.context)):
                    self.context[j] = 0
                    c_idx = j + i - b[i]
                    if max(i - b[i], 0) <= c_idx <= i + b[i] and c_idx != i and c_idx < len(subsampled_wids):
                        self.context[j] = subsampled_wids[j + i - b[i]]
                examples.append((subsampled_wids[i], self.context, self.data.get_negative_samples(subsampled_wids[i], self.ns_size)))
            return examples, len_wids

    def collate(self, list batches):
        cdef long t

        return (
            [t for b in batches for t, _, _ in b[0]],
            [c for b in batches for _, c, _ in b[0]],
            [neg for b in batches for _, _, neg in b[0]],
            sum([b[1] for b in batches]),
        )
