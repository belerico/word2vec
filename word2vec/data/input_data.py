import torch
from torch.utils.data import Dataset
import numpy as np


class InputData:
    def __init__(
        self,
        data_path: str,
        min_count=5,
        unigram_pow=0.75,
        sample_thr=0.001,
        unigram_table_size=1e8,
        max_sentence_length=1000,
    ):
        self.data_path = data_path
        self.min_count = min_count
        self.unigram_pow = unigram_pow
        self.sample_thr = sample_thr
        self.unigram_table_size = unigram_table_size
        self.max_sentence_length = max_sentence_length

        self.word_freqs = dict()
        self.word2id = dict()
        self.id2word = dict()
        self.sentence_cnt = 0
        self.word_cnt = 0
        self.unique_word_cnt = 0
        self.neg_idx = 0
        self.unigram_table = []
        self.keep_table = []
        self.sorted = []
        self.init_vocab()
        self.init_unigram_table()
        self.init_keep_table()
        # self.max_unigram_prob = np.max(self.unigram_table)

    def init_vocab(self):
        with open(self.data_path, "r") as f:
            wid = 0
            print("Building vocab")
            # Read file in chunk
            eof = False
            while not eof:
                line = f.read(self.max_sentence_length)
                if not line:
                    eof = True
                else:
                    whitespace = False
                    while not whitespace:
                        char = f.read(1)
                        if char.isspace() or not char:
                            whitespace = True
                        else:
                            line += char
                    if char:
                        self.sentence_cnt += 1
                        for w in line.strip().split():
                            if len(w) > 0:
                                self.word_freqs[w] = (
                                    self.word_freqs.get(w, 0) + 1
                                )
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
                                if self.word_cnt % 1e6 == 0:
                                    print(
                                        "Read "
                                        + str(int(self.word_cnt / 1e6))
                                        + "M words"
                                    )
                    else:
                        eof = True
        # Replace word keys with ids and
        # keep only those words with frequency >= min count
        self.word_freqs = {
            self.word2id[k]: v
            for k, v in self.word_freqs.items()
            if v >= self.min_count
        }
        # Sorted indices by frequency, descending order
        self.sorted = np.argsort(list(self.word_freqs.values()))[::-1]
        print("Done")
        print("Word count:", self.word_cnt)
        print("Sentence count:", self.sentence_cnt)
        print("Unique word count:", self.unique_word_cnt)

    def init_unigram_table(self):
        print("Building unigram table for negative sampling")
        pow_freqs = self.get_sorted_freqs() ** self.unigram_pow
        all_pow_freqs = np.sum(pow_freqs)
        count = np.round(pow_freqs / all_pow_freqs * self.unigram_table_size)
        for sorted_wid, c in enumerate(count):
            self.unigram_table += [self.sorted[sorted_wid]] * int(c)
        np.random.shuffle(self.unigram_table)
        print("Done")

    def get_sorted_freqs(self):
        return np.array(list(self.word_freqs.values()))[self.sorted]

    def init_keep_table(self):
        print("Building discard table for subsampling")
        x = np.array(list(self.word_freqs.values())) / self.word_cnt
        self.keep_table = (np.sqrt(x / self.sample_thr) + 1) * (
            self.sample_thr / x
        )
        print("Done")

    def get_negative_samples(
        self, target: int, context: int, ns_size=5, op_max=100
    ):
        neg = self.unigram_table[self.neg_idx : self.neg_idx + ns_size]
        self.neg_idx = (self.neg_idx + ns_size) % len(self.unigram_table)
        if len(neg) != ns_size:
            return np.concatenate((neg, self.unigram_table[0 : self.neg_idx]))
        return neg


class Word2vecDataset(Dataset):
    def __init__(
        self,
        data: InputData,
        window_size=5,
        ns_size=5,
        max_sentence_length=1000,
    ):
        self.data = data
        self.window_size = window_size
        self.ns_size = ns_size
        self.input_file = open(data.data_path, encoding="utf8")
        self.max_sentence_length = max_sentence_length

    def __len__(self):
        return self.data.sentence_cnt

    def __getitem__(self, idx):
        while True:
            # Reading file in chunk: max sentence length
            # as per Mikolov word2vec is 1000
            line = self.input_file.read(self.max_sentence_length)
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.read(self.max_sentence_length)
            else:
                whitespace = False
                while not whitespace:
                    char = self.input_file.read(1)
                    if char.isspace() or not char:
                        whitespace = True
                    else:
                        line += char
            if len(line) > 1:
                words = line.split()
                if len(words) > 1:
                    wids = [
                        self.data.word2id[w]
                        for w in words
                        if w in self.data.word2id
                        and np.random.rand()
                        < self.data.keep_table[self.data.word2id[w]]
                    ]
                    # Shrink window by b
                    b = np.random.randint(0, self.window_size - 1)
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
