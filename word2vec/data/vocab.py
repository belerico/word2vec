import logging
import os
import pickle
import random
import re
from collections import Counter, deque

import numpy as np

np.random.seed(42)


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
        self.train_words = 0
        self.neg_idx = 0
        self.unigram_table = []
        self.sorted = []
        self.init_vocab()
        self.init_unigram_table()
        self.unigram_table_len = len(self.unigram_table)

        logging.info("Train words: " + str(self.train_words))
        logging.info("Word (after min) count: " + str(self.word_cnt))
        logging.info("Sentences count: " + str(self.sentence_cnt))
        logging.info("Unique word count: " + str(self.unique_word_cnt))

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
        delim = re.compile(r"(\S+|\n)")
        word_freqs = Counter()
        tokens = deque()
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
                    split = delim.findall(chunk)
                    tokens.extend(split)
                    word_freqs.update(split)
        logging.info("Done")

        logging.info("Updating info")
        # Keep only those words with a frequency >= min_count
        wid = 1
        for w, c in word_freqs.items():
            self.train_words += c
            if w != "\n" and c >= self.min_count:
                self.word2id[w] = wid
                self.id2word[wid] = w
                self.word_freqs[w] = c
                self.word_cnt += c
                wid += 1
        del word_freqs
        self.unique_word_cnt = wid - 1
        logging.info("Done")

        # Create the discard probability table
        self.discard_table = np.zeros(len(self.word_freqs) + 1, dtype=np.float)
        logging.info("Building discard table for subsampling")
        for w, c in self.word_freqs.items():
            # (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
            self.discard_table[self.word2id[w]] = (
                (np.sqrt(c / (self.sample_thr * self.word_cnt)) + 1)
                * (self.sample_thr * self.word_cnt)
                / c
            )
        logging.info("Done")

        logging.info("Pre-building sentences of subsampled words")
        len_wids = 0
        sentences = []
        subsampled_wids = []
        while tokens:
            w = tokens.popleft()
            wid = self.word2id.get(w, -1)
            if wid != -1:
                len_wids += 1
                if self.discard_table[wid] < np.random.rand():
                    continue
                subsampled_wids.append(wid)
            if len(subsampled_wids) >= self.max_sentence_length or (
                w == "\n" and len(subsampled_wids) >= 1
            ):
                self.sentence_cnt += 1
                sentences.append((subsampled_wids, len_wids))
                len_wids = 0
                subsampled_wids = []
        del tokens
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

        # Sorted indices by frequency, descending order
        self.sorted = np.argsort(list(self.word_freqs.values()))[::-1]

    def init_unigram_table(self):
        logging.info("Building unigram table for negative sampling")
        pow_freqs = self.get_sorted_freqs() ** self.unigram_pow
        denom = np.sum(pow_freqs)
        count = np.round((pow_freqs / denom) * self.unigram_table_size)
        for sorted_wid, c in enumerate(count):
            self.unigram_table += [self.sorted[sorted_wid] + 1] * int(c)
        # np.random.shuffle(self.unigram_table)
        # frac = pow_freqs / denom
        # self.unigram_table = deque()
        # wid = 1  # Word ID by the descending order frequencies
        # d1 = frac[wid - 1]
        # for i in range(int(self.unigram_table_size)):
        #     self.unigram_table.append(self.sorted[wid - 1] + 1)

        #     # If the fraction of the table we have filled is greater than the
        #     # probability of choosing this word, then move to the next word.
        #     if i / self.unigram_table_size > d1:
        #         # Move to the next word.
        #         wid += 1

        #         # Calculate the probability for the new word, and accumulate it
        #         # with the probabilities of all previous words, so that we can
        #         # compare d1 to the percentage of the table that we have filled.
        #         d1 += frac[wid - 1]

        #     # Don't go past the end of the vocab.
        #     # The total weights for all words should sum up to 1,
        #     # so there shouldn't be any extra space at the end of
        #     # the table. Maybe it's possible to be off by 1, though?
        #     # Or maybe this is just precautionary.
        #     if wid >= len(pow_freqs) + 1:
        #         wid = len(pow_freqs)
        logging.info("Done")

    def get_sorted_freqs(self):
        return np.array(list(self.word_freqs.values()))[self.sorted]

    def get_negative_samples(self, target, ns_size=5):
        idx_negs = np.random.randint(
            low=0, high=len(self.unigram_table), size=ns_size
        )
        negs = [
            self.unigram_table[i]
            for i in idx_negs
            if target != self.unigram_table[i]
        ]
        return negs + [0] * (ns_size - len(negs))
