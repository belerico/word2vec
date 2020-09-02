import logging
import time

from torch.utils.data import DataLoader

from .data.dataset_c import Word2vecDataset
from .data.vocab import Vocab
from .model_c import CBOW

import cython

@cython.boundscheck(False)
@cython.wraparound(False)
class Word2Vec:

    def __init__(
        self,
        str train_file=None,
        str input_vocab_path=None,
        str output_vocab_path=None,
        str output_vec_path=None,
        str output_vec_format=None,
        str sentences_path=None,
        bint overwrite=True,
        bint sg=1,
        int emb_dimension=100,
        int min_count=5,
        int window_size=5,
        bint shrink_window_size=True,
        int ns_size=5,
        int max_sentence_length=1000,
        float unigram_pow=0.75,
        float sample_thr=0.001,
        long unigram_table_size=int(1e8),
        int epochs=10,
        float initial_lr=0.025,
        str lr_type="decay",
        bint cbow_mean=True,
    ):

        if str.lower(lr_type) not in [
            "triangular",
            "decay",
            "traingular_decay",
        ]:
            raise NotImplementedError(
                "'lr_type' must be 'triangular', 'triangular_decay' or 'decay'"
            )
        self.lr_type = str.lower(lr_type)

        if input_vocab_path:
            self.data = Vocab.load_vocab(input_vocab_path)
        else:
            self.data = Vocab(
                train_file=train_file,
                sentences_path=sentences_path,
                min_count=min_count,
                max_sentence_length=max_sentence_length,
                unigram_pow=unigram_pow,
                unigram_table_size=unigram_table_size,
                sample_thr=sample_thr,
                overwrite=overwrite,
            )
            if output_vocab_path and not input_vocab_path:
                self.data.save_vocab(output_vocab_path)

        self.dataset = Word2vecDataset(
            self.data,
            sg=sg,
            window_size=window_size,
            ns_size=ns_size,
            shrink_window_size=shrink_window_size,
            sentences_path=sentences_path,
        )

        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, collate_fn=self.dataset.collate)

        self.output_vec_path = output_vec_path
        self.output_vec_format = output_vec_format
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.overwrite = overwrite
        if not sg and initial_lr == 0.025:
            self.initial_lr = 0.05  # Default LR for CBOW

        self.model = CBOW(self.emb_size, self.emb_dimension, cbow_mean)

    def _get_lr_decay(self):
        if self.lr_type == "triangular":
            return lambda long actual_word_cnt, int epoch: self.initial_lr * (
                1.0 - actual_word_cnt / (self.data.word_cnt + 1)
            )
        elif self.lr_type == "traingular_decay":
            return lambda long actual_word_cnt, int epoch: self.initial_lr * (
                1.0 - actual_word_cnt / ((epoch + 1) * self.data.word_cnt + 1)
            )
        else:
            return lambda long actual_word_cnt, int epoch: self.initial_lr * (
                1.0 - actual_word_cnt / (self.epochs * self.data.word_cnt + 1)
            )

    def train(self):

        cdef float lr = self.initial_lr
        lr_decay = self._get_lr_decay()

        # Global word count
        cdef int i, epoch
        cdef tuple examples
        cdef double t0
        cdef long word_cnt = 0, actual_word_cnt = 0, epoch_word_cnt = 0

        for epoch in range(self.epochs):
            t0 = time.time()
            epoch_word_cnt = 0
            if self.lr_type == "triangular":
                actual_word_cnt = 0

            for examples in self.dataloader:
                self.model.forward(examples[0], examples[1], examples[2], lr)

                word_cnt += examples[3]
                epoch_word_cnt += examples[3]
                actual_word_cnt += examples[3]

                if word_cnt > 10000:
                    word_cnt = word_cnt - 10000
                    lr = lr_decay(actual_word_cnt, epoch=epoch)
                    if lr <= self.initial_lr * 0.0001:
                        lr = self.initial_lr * 0.0001

                    logging.info(
                        "Progress: {:.4f}%, Elapsed: {:.2f}s, Lr: {}".format(
                            ((epoch_word_cnt / self.data.word_cnt) * 100),
                            time.time() - t0,
                            round(lr, 8),
                        )
                    )

            logging.info(
                "Epoch: {}, Elapsed: {:.2f}s, Lr: {}".format(
                    epoch,
                    time.time() - t0,
                    round(lr, 8),
                )
            )
        if self.output_vec_path:
            self.model.save_embeddings(
                self.data.id2word,
                self.output_vec_path,
                vec_format=self.output_vec_format,
                overwrite=self.overwrite,
            )
