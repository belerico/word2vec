import logging
import os
import pickle
import string
import threading
from collections import Counter
from queue import Queue

import numpy as np


class Producer(threading.Thread):
    def __init__(
        self,
        filename: str,
        q: Queue,
        word_freqs: Counter,
        chunk_size=32768,
        simple_preprocess=True,
    ):
        super(Producer, self).__init__()
        self.file = open(filename, "r")
        self.q = q
        self.word_freqs = word_freqs
        assert chunk_size > 0
        self.chunk_size = chunk_size
        self.simple_preprocess = simple_preprocess
        if self.simple_preprocess:
            self.remove_punct = str.maketrans(
                string.punctuation, " " * len(string.punctuation)
            )

    def run(self):
        stop = False
        while not stop:
            chunk = self.file.read(self.chunk_size)
            if not chunk:
                self.q.put(-1)
                stop = True
            else:
                while True:
                    c = self.file.read(1)
                    if not c or c.isspace():
                        break
                    else:
                        chunk += c
                if self.simple_preprocess:
                    chunk = str.lower(chunk).translate(self.remove_punct)
                self.q.put(chunk)


class Consumer(threading.Thread):
    def __init__(
        self, q: Queue, word_freqs: Counter,
    ):
        super(Consumer, self).__init__()
        self.q = q
        self.word_freqs = word_freqs

    def run(self):
        stop = False
        while not stop:
            chunk = self.q.get()
            if chunk == -1:
                stop = True
            else:
                self.word_freqs.update(chunk.split())


class Vocab:
    def __init__(
        self,
        train_file=None,
        min_count=5,
        unigram_pow=0.75,
        sample_thr=0.001,
        unigram_table_size=1e8,
        max_sentence_length=1000,
        overwrite=True,
        chunk_size=32768,
        queue_buf_size=100000,
        simple_preprocess=True,
    ):
        if not train_file:
            raise FileNotFoundError("Train file path not specified")

        self.train_file = train_file
        self.min_count = min_count
        self.unigram_pow = unigram_pow
        self.sample_thr = sample_thr
        self.unigram_table_size = unigram_table_size
        self.max_sentence_length = max_sentence_length
        self.overwrite = overwrite
        self.chunk_size = chunk_size
        self.queue_buf_size = queue_buf_size
        self.simple_preprocess = simple_preprocess

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
        if not os.path.exists(vocab_path) or self.overwrite:
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

    def init_vocab(self):
        logging.info("Building vocab")
        # Start Producer-Consumer thread to read "efficiently" (I hope) the train file
        q = Queue(self.queue_buf_size)
        word_freqs = Counter()

        producer = Producer(
            self.train_file,
            q,
            word_freqs=word_freqs,
            chunk_size=self.chunk_size,
            simple_preprocess=self.simple_preprocess,
        )
        consumer = Consumer(q, word_freqs)

        producer.start()
        consumer.start()
        producer.join()
        consumer.join()
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

        logging.info("Word (after min) count: " + str(self.word_cnt))
        logging.info("Unique word count: " + str(self.unique_word_cnt))

        # Sorted indices by frequency, descending order
        self.sorted = np.argsort(list(self.word_freqs.values()))[::-1]

    def init_unigram_table(self):
        logging.info("Building unigram table for negative sampling")
        pow_freqs = self.get_sorted_freqs() ** self.unigram_pow
        all_pow_freqs = np.sum(pow_freqs)
        count = np.round(pow_freqs / all_pow_freqs * self.unigram_table_size)
        for sorted_wid, c in enumerate(count):
            self.unigram_table += [self.sorted[sorted_wid] + 1] * int(round(c))
        np.random.shuffle(self.unigram_table)
        logging.info("Done")

    def get_sorted_freqs(self):
        return np.array(list(self.word_freqs.values()))[self.sorted]

    def get_negative_samples(self, ns_size=5):
        neg = self.unigram_table[self.neg_idx : self.neg_idx + ns_size]
        self.neg_idx = (self.neg_idx + ns_size) % len(self.unigram_table)
        if len(neg) != ns_size:
            return np.concatenate((neg, self.unigram_table[0 : self.neg_idx]))
        return neg
