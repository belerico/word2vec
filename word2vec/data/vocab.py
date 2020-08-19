import logging
import marshal
import os
import pickle

import numpy as np


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
        self.init_vocab(sentences_path, overwrite=overwrite)
        self.init_unigram_table()

        # Add padding index
        self.id2word[0] = "PAD"
        self.word2id["PAD"] = 0
        self.word_freqs[0] = 0

    def save_vocab(self, vocab_path, overwrite=True):
        if not os.path.exists(vocab_path) or overwrite:
            if not os.path.exists(os.path.dirname(vocab_path)):
                os.makedirs(os.path.dirname(vocab_path))
            logging.info("Saving vocab to " + vocab_path)
            pickle.dump(self, open(os.path.join(vocab_path), "wb"), protocol=-1)
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

    def init_vocab(self, sentences_path: str, overwrite=False):
        with open(self.train_file, "r") as f:
            eof = False
            train_words = 0
            word_freqs = dict()
            sentences = []

            logging.info("Building vocab")
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
                        if char.isspace():
                            whitespace = True
                        elif char == "":
                            eof = True
                            whitespace = True
                        else:
                            line += char

                if line != "\n" and line != "":
                    words = line.strip().split()
                    # Collect infos only if in a sentence there're at least 2 words
                    if len(words) > 1:
                        sentence = []
                        for w in words:
                            if len(w) > 0:
                                word_freqs[w] = word_freqs.get(w, 0) + 1
                                train_words += 1
                                sentence.append(w)
                                if train_words % 1e6 == 0 and train_words >= 1e6:
                                    logging.info(
                                        "Read " + str(int(train_words / 1e6)) + "M words"
                                    )
                        sentences.append(sentence)
            logging.info("Done")

        logging.info("Updating info")
        # Keep only those words with a frequency >= min_count
        wid = 1
        for w, c in word_freqs.items():
            if c >= self.min_count:
                self.word2id[w] = wid
                self.id2word[wid] = w
                self.word_freqs[w] = c
                wid += 1
        del word_freqs
        self.unique_word_cnt = wid - 1
        logging.info("Done")

        logging.info("Building sentences")
        updated_sentences = []
        for i, sentence in enumerate(sentences):
            updated_sentences.append(
                [self.word2id[w] for w in sentence if w in self.word2id]
            )
            self.word_cnt += len(updated_sentences[-1])
            if updated_sentences[-1]:
                self.sentence_cnt += 1
        del sentences
        logging.info("Done")

        if not os.path.exists(sentences_path) or overwrite:
            if not os.path.exists(os.path.dirname(sentences_path)):
                os.makedirs(os.path.dirname(sentences_path))
            logging.info("Saving sentences (incrementally) to " + sentences_path)
            with open(os.path.join(sentences_path), "wb", 1024 * 1024) as f:
                for sentence in updated_sentences:
                    if sentence:
                        marshal.dump(sentence, f, )
            del updated_sentences
            logging.info("Done")
        else:
            raise FileExistsError("'" + sentences_path + "' already exists")

        # Create the discard probability table
        self.discard_table = [0]
        logging.info("Building discard table for subsampling")
        for _, c in self.word_freqs.items():
            f = c / self.word_cnt
            self.discard_table.append(
                (np.sqrt(f / self.sample_thr) + 1) * (self.sample_thr / f)
            )
        logging.info("Done")

        logging.info("Train words: " + str(train_words))
        logging.info("Word (after min) count: " + str(self.word_cnt))
        logging.info("Sentence count: " + str(self.sentence_cnt))
        logging.info("Unique word count: " + str(self.unique_word_cnt))

        # Sorted indices by frequency, descending order
        self.sorted = np.argsort(list(self.word_freqs.values()))[::-1]

    def init_unigram_table(self):
        logging.info("Building unigram table for negative sampling")
        pow_freqs = self.get_sorted_freqs() ** self.unigram_pow
        all_pow_freqs = np.sum(pow_freqs)
        count = np.round(pow_freqs / all_pow_freqs * self.unigram_table_size)
        for sorted_wid, c in enumerate(count):
            self.unigram_table += [self.sorted[sorted_wid] + 1] * int(c)
        np.random.shuffle(self.unigram_table)
        logging.info("Done")

    def get_sorted_freqs(self):
        return np.array(list(self.word_freqs.values()))[self.sorted]

    def init_discard_table(self):
        logging.info("Building discard table for subsampling")
        x = np.array(list(self.word_freqs.values())) / self.word_cnt
        self.discard_table = (np.sqrt(x / self.sample_thr) + 1) * (self.sample_thr / x)
        logging.info("Done")

    def get_negative_samples(self, ns_size=5):
        neg = self.unigram_table[self.neg_idx : self.neg_idx + ns_size]
        self.neg_idx = (self.neg_idx + ns_size) % len(self.unigram_table)
        if len(neg) != ns_size:
            return np.concatenate((neg, self.unigram_table[0 : self.neg_idx]))
        return neg
