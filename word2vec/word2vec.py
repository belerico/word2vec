import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from word2vec.data.input_data import InputData
from word2vec.data.dataset import Word2vecDataset
from word2vec.model import SkipGram


class Word2Vec:
    def __init__(
        self,
        train_file=None,
        output_vocab_dir=None,
        output_vec_file=None,
        emb_dimension=100,
        batch_size=1,
        min_count=5,
        window_size=5,
        ns_size=5,
        epochs=10,
        initial_lr=0.001,
    ):

        self.data = InputData(train_file, min_count)
        if output_vocab_dir:
            self.data.save_vocab(output_vocab_dir)

        dataset = Word2vecDataset(
            self.data, window_size=window_size, ns_size=ns_size,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate,
        )

        self.output_vec_file = output_vec_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.model = SkipGram(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr)
        for epoch in range(self.epochs):

            running_loss = 0.0
            word_cnt = 0
            t0 = time.time()

            for i, sample_batched in enumerate(self.dataloader):
                if len(sample_batched[0]) > 0:

                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    optimizer.zero_grad()
                    loss = self.model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    word_cnt += len(sample_batched[0])

                    if word_cnt > 10000:
                        word_cnt = word_cnt - 10000
                        lr = self.initial_lr * (
                            1.0 - (i + 1) / self.data.sentence_cnt
                        )
                        if lr >= self.initial_lr * 0.0001:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = lr

                        print(
                            "Processed sentences: {:.4f}%, Elapsed: {:.2f}s".format(
                                ((i / self.data.sentence_cnt) * 100),
                                time.time() - t0,
                            )
                        )

            epoch_loss = running_loss / len(self.dataloader)
            print(
                "Epoch: {}, Elapsed: {:.2f}s, Training Loss: {:.4f}".format(
                    epoch, time.time() - t0, epoch_loss
                )
            )
