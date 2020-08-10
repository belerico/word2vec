import os

import numpy as np


class InputData:
    def __init__(
        self, data_path: str, min_count=5, unigram_pow=0.75, sample_thr=0.001
    ):
        self.data_path = data_path
        self.min_count = min_count
        self.unigram_pow = unigram_pow
        self.sample_thr = sample_thr
        self.word_freqs = dict()
        self.word2id = dict()
        self.id2word = dict()
        self.sentence_cnt = 0
        self.word_cnt = 0
        self.unique_word_cnt = 0
        self.unigram_table = []
        self.keep_table = []
        self.sorted = []

    def init_vocab(self):
        with open(self.data_path, "r") as f:
            wid = 0
            for line in f:
                if len(line) > 1:
                    self.sentence_cnt += 1
                    for w in line.strip().split():
                        if len(w) > 0:
                            self.word_freqs[w] = self.word_freqs.get(w, 0) + 1
                            self.word_cnt += 1
                            # Update stats only for words that has a frequency
                            # greater than min_count
                            if self.word_freqs[w] >= self.min_count:
                                # If it's the "first" time we encounter word w
                                if self.word_freqs[w] == self.min_count:
                                    self.id2word[wid] = w
                                    self.word2id[w] = wid
                                    self.unique_word_cnt += 1
                                    wid += 1
                            if self.word_cnt % 1000 == 0:
                                print(
                                    "Read "
                                    + str(int(self.word_cnt / 1000))
                                    + "K words."
                                )
        # Replace word keys with ids and
        # keep only those words with frequency >= min count
        self.word_freqs = {
            self.word2id[k]: v
            for k, v in self.word_freqs.items()
            if v >= self.min_count
        }
        # Sorted indices by frequency, descending order
        self.sorted = np.argsort(list(self.word_freqs.values()))[::-1]

    def init_unigram_table(self):
        pow_freqs = (
            np.array(list(self.word_freqs.values())) ** self.unigram_pow
        )
        all_pow_freqs = np.sum(pow_freqs)
        self.unigram_table = pow_freqs / all_pow_freqs

    def get_sorted_freqs(self):
        return np.array(list(self.word_freqs.values()))[self.sorted]

    def init_keep_table(self):
        x = np.array(list(self.word_freqs.values())) / self.word_cnt
        self.keep_table = (np.sqrt(x / self.sample_thr) + 1) * (
            self.sample_thr / x
        )
