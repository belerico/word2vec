import os
import pickle

import numpy as np


class Vocab:
    def __init__(
        self,
        train_file=None,
        min_count=5,
        unigram_pow=0.75,
        sample_thr=0.001,
        unigram_table_size=1e8,
        max_sentence_length=1000,
    ):
        if not train_file:
            raise FileNotFoundError("Train file path not specified")

        self.train_file = train_file
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
        self.sorted = []
        self.init_vocab()
        self.init_unigram_table()

        # Add padding index
        self.id2word[0] = "PAD"
        self.word2id["PAD"] = 0
        self.word_freqs[0] = 0

    def save_vocab(self, vocab_path):
        if not os.path.exists(vocab_path):
            if not os.path.exists(os.path.dirname(vocab_path)):
                os.makedirs(os.path.dirname(vocab_path))
            print("Saving vocab to " + vocab_path)
            pickle.dump(
                self, open(os.path.join(vocab_path), "wb"),
            )
            print("Done")
        else:
            raise FileExistsError("'" + vocab_path + "' already exists")

    @staticmethod
    def load_vocab(vocab_path):
        if os.path.exists(vocab_path):
            print("Loading vocab from " + vocab_path)
            obj = pickle.load(open(vocab_path, "rb"))
            print("Done")
            return obj
        else:
            raise FileNotFoundError("'" + vocab_path + "' not found")

    def init_vocab(self):
        with open(self.train_file, "r") as f:
            eof = False
            wid = 1
            word_freqs = dict()

            print("Building vocab")
            while not eof:
                char_read = 0
                new_line = False
                line = ""

                # Read file in chunk or until a new line
                while not eof and char_read < self.max_sentence_length and not new_line:
                    char = f.read(1)
                    if char == "\n":
                        new_line = True
                    elif char == "":
                        eof = True
                    else:
                        line += char
                        char_read += 1

                if not new_line:
                    # If a word is truncated after "max_sentence_length" chars,
                    # read until any whitespace is found
                    whitespace = False
                    while not whitespace:
                        char = f.read(1)
                        if char.isspace() or not char:
                            whitespace = True
                        else:
                            line += char

                if line != "\n" and line != "":
                    if len(line) > 1:
                        self.sentence_cnt += 1
                    for w in line.strip().split():
                        if len(w) > 0:
                            word_freqs[w] = word_freqs.get(w, 0) + 1
                            if word_freqs[w] >= self.min_count:
                                self.word_cnt += 1
                                if word_freqs[w] == self.min_count:
                                    # Update stats only for words that has a frequency
                                    # greater than min_count
                                    self.id2word[wid] = w
                                    self.word2id[w] = wid
                                    self.unique_word_cnt += 1
                                    wid += 1
                                self.word_freqs[self.word2id[w]] = word_freqs[w]
                            if self.word_cnt % 1e6 == 0 and self.word_cnt > 1e6:
                                print("Read " + str(int(self.word_cnt / 1e6)) + "M words")
            print("Done")

        # Create the discard probability table
        self.discard_table = [0]
        print("Building discard table for subsampling")
        for _, c in self.word_freqs.items():
            f = c / self.word_cnt
            self.discard_table.append(
                (np.sqrt(f / self.sample_thr) + 1) * (self.sample_thr / f)
            )
        print("Done")
        
        print("Word count:", self.word_cnt)
        print("Sentence count:", self.sentence_cnt)
        print("Unique word count:", self.unique_word_cnt)

        # Sorted indices by frequency, descending order
        self.sorted = np.argsort(list(self.word_freqs.values()))[::-1]

    def init_unigram_table(self):
        print("Building unigram table for negative sampling")
        pow_freqs = self.get_sorted_freqs() ** self.unigram_pow
        all_pow_freqs = np.sum(pow_freqs)
        count = np.round(pow_freqs / all_pow_freqs * self.unigram_table_size)
        for sorted_wid, c in enumerate(count):
            self.unigram_table += [self.sorted[sorted_wid] + 1] * int(c)
        np.random.shuffle(self.unigram_table)
        print("Done")

    def get_sorted_freqs(self):
        return np.array(list(self.word_freqs.values()))[self.sorted]

    def init_discard_table(self):
        print("Building discard table for subsampling")
        x = np.array(list(self.word_freqs.values())) / self.word_cnt
        self.discard_table = (np.sqrt(x / self.sample_thr) + 1) * (self.sample_thr / x)
        print("Done")

    def get_negative_samples(self, ns_size=5):
        neg = self.unigram_table[self.neg_idx : self.neg_idx + ns_size]
        self.neg_idx = (self.neg_idx + ns_size) % len(self.unigram_table)
        if len(neg) != ns_size:
            return np.concatenate((neg, self.unigram_table[0 : self.neg_idx]))
        return neg
