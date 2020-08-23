import logging
import os
import pickle
import threading
from collections import Counter
from queue import Queue

import numpy as np


class Producer(threading.Thread):
    def __init__(
        self,
        filename: str,
        q: Queue,
        sentences: list,
        word_freqs: Counter,
        chunk_size=32768,
        max_sentence_length=1000,
    ):
        super(Producer, self).__init__()
        self.file = open(filename, "r")
        self.q = q
        self.sentences = sentences
        self.word_freqs = word_freqs
        assert chunk_size > 0
        self.chunk_size = chunk_size
        assert isinstance(max_sentence_length, int) and max_sentence_length > 0
        self.max_sentence_length = max_sentence_length

    def run(self):
        stop = False
        while not stop:
            chunk = self.file.read(self.chunk_size)
            if not chunk:
                self.q.put(-1)
                stop = True
            else:
                for sentence in chunk.split("\n"):
                    if len(sentence) <= self.max_sentence_length:
                        self.sentences.append(sentence)
                        self.word_freqs.update(sentence.split())
                    else:
                        self.q.put(sentence)


class Consumer(threading.Thread):
    def __init__(
        self, q: Queue, sentences: str, word_freqs: Counter, max_sentence_length=1000
    ):
        super(Consumer, self).__init__()
        self.buffer = []
        self.q = q
        self.sentences = sentences
        self.word_freqs = word_freqs
        self.max_sentence_length = max_sentence_length

    def run(self):
        stop = False
        while not stop:
            sentence = self.q.get()
            if sentence == -1:
                stop = True
            else:
                self.buffer.append(sentence)
                self.q.task_done()
                while len(self.buffer) > 1 or len(self.buffer[0]) > 1000:
                    i = 0
                    sent = self.buffer.pop(0)
                    while (
                        i + self.max_sentence_length < len(sent)
                        and not sent[self.max_sentence_length + i].isspace()
                    ):
                        i += 1
                    if i + self.max_sentence_length < len(sent):
                        new_sentence = sent[: self.max_sentence_length + i].split()
                        self.sentences.append(new_sentence)
                        self.word_freqs.update(new_sentence)
                        self.buffer.insert(0, sent[self.max_sentence_length + i :])
                    elif self.buffer:
                        self.buffer.insert(0, sent + self.buffer.pop(0))
                    else:
                        self.buffer.insert(0, sent)
                        break

        while len(self.buffer) > 1 or len(self.buffer[0]) > 1000:
            i = 0
            sent = self.buffer.pop(0)
            while (
                i + self.max_sentence_length < len(sent)
                and not sent[self.max_sentence_length + i].isspace()
            ):
                i += 1
            if i + self.max_sentence_length < len(sent):
                new_sentence = sent[: self.max_sentence_length + i].split()
                self.sentences.append(new_sentence)
                self.word_freqs.update(new_sentence)
                self.buffer.insert(0, sent[self.max_sentence_length + i :])
            elif self.buffer:
                self.buffer.insert(0, sent + self.buffer.pop(0))
            else:
                self.buffer.insert(0, sent)
                break

        self.sentences.append(self.buffer[0] if self.buffer else "")
        self.word_freqs.update(self.buffer[0].split() if self.buffer else "")


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
        queue_buf_size=100000,
    ):
        if not train_file:
            raise FileNotFoundError("Train file path not specified")

        self.train_file = train_file
        self.min_count = min_count
        self.unigram_pow = unigram_pow
        self.sample_thr = sample_thr
        self.unigram_table_size = unigram_table_size
        self.max_sentence_length = max_sentence_length
        self.chunk_size = chunk_size
        self.queue_buf_size = queue_buf_size

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
        logging.info("Building vocab")
        # Start Producer-Consumer thread to read "efficiently" (I hope) the train file
        q = Queue(self.queue_buf_size)
        word_freqs = Counter()
        sentences = []

        producer = Producer(self.train_file, q, sentences, word_freqs)
        consumer = Consumer(q, sentences, word_freqs)

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
                wid += 1
        del word_freqs
        self.unique_word_cnt = wid - 1
        logging.info("Done")

        if not os.path.exists(sentences_path) or overwrite:
            if not os.path.exists(os.path.dirname(sentences_path)):
                os.makedirs(os.path.dirname(sentences_path))
            logging.info(
                "Building and saving sentences (incrementally) to " + sentences_path
            )
            with open(os.path.join(sentences_path), "wb") as f:
                s = []
                for i, sentence in enumerate(sentences):
                    s = [self.word2id[w] for w in sentence if w in self.word2id]
                    if s:
                        self.word_cnt += len(s)
                        self.sentence_cnt += 1
                        pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)
            del sentences
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

        # logging.info("Train words: " + str(train_words))
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
