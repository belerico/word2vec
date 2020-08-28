import logging
import os
import pickle
import random
from collections import Counter

import numpy as np

random.seed(42)


class Vocab:
    def __init__(
        self,
        train_file=None,
        sentences_path=None,
        min_count=5,
        unigram_pow=0.75,
        sample_thr=0.001,
        unigram_table_size=1e8,
        max_sentence_length=1000,
        overwrite=True,
        chunk_size=32768,
    ):
        if not train_file:
            raise FileNotFoundError("Train file path not specified")

        self.train_file = train_file
        self.sentences_path = sentences_path
        self.min_count = min_count
        self.unigram_pow = unigram_pow
        self.sample_thr = sample_thr
        self.unigram_table_size = unigram_table_size
        self.max_sentence_length = max_sentence_length
        self.overwrite = overwrite
        self.chunk_size = chunk_size

        self.word_freqs = dict()
        self.word2id = dict()
        self.id2word = dict()
        self.sentence_cnt = 0
        self.word_cnt = 0
        self.unique_word_cnt = 0
        self.neg_idx = 0
        self.unigram_table = []
        self.sorted = []
        self.init_vocab()
        self.init_unigram_table()
        self.unigram_table_len = len(self.unigram_table)

        # Add padding index
        self.id2word[0] = "PAD"
        self.word2id["PAD"] = 0
        self.word_freqs[0] = 0

    def save_vocab(self, vocab_path):
        if not os.path.exists(vocab_path) or self.overwrite:
            if not os.path.exists(os.path.dirname(vocab_path)):
                os.makedirs(os.path.dirname(vocab_path))
            logging.info("Saving vocab to " + vocab_path)
            pickle.dump(
                self,
                open(os.path.join(vocab_path), "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            logging.info("Done")
        else:
            raise FileExistsError("'" + vocab_path + "' already exists")

    @staticmethod
    def load_vocab(vocab_path):
        if os.path.exists(vocab_path):
            logging.info("Loading vocab from " + vocab_path)
            obj = pickle.load(open(vocab_path, "rb"))
            logging.info("Done")
            return obj
        else:
            raise FileNotFoundError("'" + vocab_path + "' not found")

    def init_vocab(self):
        logging.info("Building vocab")
        word_freqs = Counter()
        with open(self.train_file, "r") as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                else:
                    while True:
                        c = f.read(1)
                        if not c or c.isspace():
                            break
                        else:
                            chunk += c
                    word_freqs.update(chunk.split())
        logging.info("Done")

        logging.info("Updating info")
        # Keep only those words with a frequency >= min_count
        wid = 1
        for w, c in word_freqs.items():
            if c >= self.min_count:
                self.word2id[w] = wid
                self.id2word[wid] = w
                self.word_freqs[w] = c
                self.word_cnt += c
                wid += 1
        del word_freqs
        self.unique_word_cnt = wid - 1
        logging.info("Done")

        # Create the discard probability table
        self.discard_table = [0]
        logging.info("Building discard table for subsampling")
        for c in self.word_freqs.values():
            f = c / self.word_cnt
            self.discard_table.append(
                (np.sqrt(f / self.sample_thr) + 1) * (self.sample_thr / f)
            )
        logging.info("Done")

        logging.info("Pre-building sentences of subsampled words")
        w = ""
        len_wids = 0
        sentences = []
        subsampled_wids = []
        with open(self.train_file, "r") as f:
            while True:
                c = f.read(1)
                if c.isspace() or not c:
                    wid = self.word2id.get(w, -1)
                    if wid != -1:
                        len_wids += 1
                        if self.discard_table[wid] >= random.random():
                            subsampled_wids.append(wid)
                    if len(subsampled_wids) >= self.max_sentence_length or (
                        (c == "\n" or not c) and len(subsampled_wids) > 1
                    ):
                        self.sentence_cnt += 1
                        sentences.append((subsampled_wids, len_wids))
                        len_wids = 0
                        subsampled_wids = []
                        if not c:
                            break
                    w = ""
                else:
                    w += c
        logging.info("Done")
        if not os.path.exists(self.sentences_path) or self.overwrite:
            if not os.path.exists(os.path.dirname(self.sentences_path)):
                os.makedirs(os.path.dirname(self.sentences_path))
            logging.info("Saving sentences to " + self.sentences_path)
            pickle.dump(
                sentences,
                open(self.sentences_path, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            logging.info("Done")
        else:
            raise FileExistsError(
                "'" + self.sentences_path + "' already exists"
            )
        del sentences, subsampled_wids

        logging.info("Word (after min) count: " + str(self.word_cnt))
        logging.info("Sentences count: " + str(self.sentence_cnt))
        logging.info("Unique word count: " + str(self.unique_word_cnt))

        # Sorted indices by frequency, descending order
        self.sorted = np.argsort(list(self.word_freqs.values()))[::-1]

    def init_unigram_table(self):
        logging.info("Building unigram table for negative sampling")
        pow_freqs = self.get_sorted_freqs() ** self.unigram_pow
        all_pow_freqs = np.sum(pow_freqs)
        count = (pow_freqs / all_pow_freqs) * self.unigram_table_size
        for sorted_wid, c in enumerate(count):
            self.unigram_table += [self.sorted[sorted_wid] + 1] * int(round(c))
        np.random.shuffle(self.unigram_table)
        logging.info("Done")

    def get_sorted_freqs(self):
        return np.array(list(self.word_freqs.values()))[self.sorted]

    def get_negative_samples(self, ns_size=5):
        neg = self.unigram_table[self.neg_idx : self.neg_idx + ns_size]
        self.neg_idx = (self.neg_idx + ns_size) % self.unigram_table_len
        if len(neg) != ns_size:
            return neg + self.unigram_table[0 : self.neg_idx]
        return neg
