import time
from word2vec.data.dataset import Word2vecDataset
from word2vec.data.vocab import Vocab
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from word2vec.model import SkipGram, CBOW


class Word2Vec:
    def __init__(
        self,
        train_file=None,
        input_vocab_path=None,
        output_vocab_path=None,
        output_vec_path=None,
        output_vec_format=None,
        sentences_path=None,
        overwrite=True,
        sg=1,
        emb_dimension=100,
        batch_size=1,
        min_count=5,
        window_size=5,
        shrink_window_size=True,
        ns_size=5,
        max_sentence_length=1000,
        unigram_pow=0.75,
        sample_thr=0.001,
        unigram_table_size=1e8,
        epochs=10,
        initial_lr=0.025,
        lr_type="decay",
        optim="sgd",
        cbow_mean=True,
        mikolov_context=True,
        use_gpu=1,
    ):

        if str.lower(lr_type) not in ["triangular", "decay", "traingular_decay"]:
            raise NotImplementedError(
                "'lr_type' must be 'triangular', 'triangular_decay' or 'decay'"
            )
        self.lr_type = str.lower(lr_type)

        if str.lower(optim) not in ["sgd", "adam"]:
            raise NotImplementedError("'optim' must be 'sgd' or 'adam'")
        self.optim = str.lower(optim)

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
            )
            if output_vocab_path and not input_vocab_path:
                self.data.save_vocab(output_vocab_path)

        dataset = Word2vecDataset(
            self.data,
            sg=sg,
            window_size=window_size,
            ns_size=ns_size,
            shrink_window_size=shrink_window_size,
            sentences_path=sentences_path,
            mikolov_context=mikolov_context,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_sg if sg else dataset.collate_cw,
        )

        self.output_vec_path = output_vec_path
        self.output_vec_format = output_vec_format
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.overwrite = overwrite
        if not sg and initial_lr == 0.025:
            self.initial_lr = 0.05  # Default LR for CBOW

        self.model = (
            SkipGram(self.emb_size, self.emb_dimension)
            if sg
            else CBOW(self.emb_size, self.emb_dimension, cbow_mean)
        )

        self.use_gpu = torch.cuda.is_available() and use_gpu
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        if self.use_gpu:
            self.model.cuda()

    def train(self):
        if self.optim == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr)
        else:
            optimizer = optim.SparseAdam(self.model.parameters(), lr=self.initial_lr)
        lr = self.initial_lr

        # Global running loss and word count
        running_loss = 0.0
        word_cnt = 0
        actual_word_cnt = 0

        for epoch in range(self.epochs):
            t0 = time.time()
            if self.lr_type == "triangular":
                actual_word_cnt = 0

            for i, sample_batched in enumerate(self.dataloader):
                if (
                    len(sample_batched)
                    and len(sample_batched[0]) > 0
                    and len(sample_batched[1]) > 0
                ):

                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    optimizer.zero_grad()
                    loss = self.model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    word_cnt += sample_batched[3]
                    actual_word_cnt += sample_batched[3]

                    if word_cnt > 10000:
                        word_cnt = word_cnt - 10000
                        if self.lr_type == "triangular":
                            lr = self.initial_lr * (
                                1.0 - actual_word_cnt / (self.data.word_cnt + 1)
                            )
                        elif self.lr_type == "traingular_decay":
                            lr = self.initial_lr * (
                                1.0
                                - actual_word_cnt / ((epoch + 1) * self.data.word_cnt + 1)
                            )
                        else:
                            lr = self.initial_lr * (
                                1.0
                                - actual_word_cnt / (self.epochs * self.data.word_cnt + 1)
                            )
                        if lr <= self.initial_lr * 0.0001:
                            lr = self.initial_lr * 0.0001
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

                    if i % 200 == 0:
                        logging.info(
                            "Progress: {:.4f}%, Elapsed: {:.2f}s, Lr: {}, Loss: {:.4f}".format(
                                ((i / self.data.sentence_cnt) * 100),
                                time.time() - t0,
                                round(lr, 8),
                                running_loss / (actual_word_cnt),
                            )
                        )
            else:
                logging.info("Empty batch: maybe next time")
                word_cnt += sample_batched[3]
                actual_word_cnt += sample_batched[3]

            logging.info(
                "Epoch: {}, Elapsed: {:.2f}s, Training Loss: {:.4f}".format(
                    epoch, time.time() - t0, running_loss / actual_word_cnt
                )
            )
        if self.output_vec_path:
            self.model.save_embeddings(
                self.data.id2word,
                self.output_vec_path,
                vec_format=self.output_vec_format,
                overwrite=self.overwrite,
            )

