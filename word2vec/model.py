import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# from opt_einsum import contract


class Word2Vec(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(Word2Vec, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension

        # syn0: embedding for input words
        # syn1: embedding for output words
        self.syn0 = nn.Embedding(
            emb_size, emb_dimension, sparse=True, padding_idx=0
        )
        self.syn1 = nn.Embedding(
            emb_size, emb_dimension, sparse=True, padding_idx=0
        )

        init_range = 0.5 / self.emb_dimension
        init.uniform_(self.syn0.weight.data, -init_range, init_range)
        init.constant_(self.syn1.weight.data, 0)
        self.syn0.weight.data[0, :] = 0

    def forward(self, pos_u, pos_v, neg_v):
        raise NotImplementedError

    def save_embeddings(
        self, id2word, output_vec_path: str, vec_format="txt", overwrite=True
    ):
        assert vec_format in ["txt", "pkl"]
        if not os.path.exists(os.path.dirname(output_vec_path)):
            os.makedirs(os.path.dirname(output_vec_path))
        embs = self.syn0.weight.cpu().data.numpy()
        output_vec_path = os.path.splitext(output_vec_path)[0]
        if vec_format is None or vec_format == "txt":
            if not os.path.exists(output_vec_path + ".txt") or overwrite:
                print("Save embeddings to " + output_vec_path + ".txt")
                with open(output_vec_path + ".txt", "w") as f:
                    f.write("%d %d\n" % (len(id2word), self.emb_dimension))
                    for wid, w in id2word.items():
                        e = " ".join(map(lambda x: str(x), embs[wid]))
                        f.write("%s %s\n" % (w, e))
            else:
                raise FileExistsError(
                    "'" + output_vec_path + ".txt' already exists"
                )
        else:
            if not os.path.exists(output_vec_path + ".pkl") or overwrite:
                print("Save embeddings to " + output_vec_path + ".pkl")
                embs_tmp = {w: embs[wid] for wid, w in id2word.items()}
                pickle.dump(
                    embs_tmp, open(output_vec_path + ".pkl", "wb"),
                )
            else:
                raise FileExistsError(
                    "'" + output_vec_path + ".pkl' already exists"
                )
        print("Done")


class SkipGram(Word2Vec):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGram, self).__init__(emb_size, emb_dimension)

    def forward(self, target, context, negatives):
        t = self.syn1(target)
        c = self.syn0(context)
        n = self.syn1(negatives)

        # pos_scores = (u_embs * self.v_embs(pos_v)).sum(dim=1)
        pos_scores = torch.mul(c, t)
        pos_scores.register_hook(lambda x: x.clamp(min=-10, max=10))
        pos_scores = torch.sum(pos_scores, dim=1)
        # pos_scores = torch.einsum("ij,ij->i", [u_embs, self.v_embs(pos_v)])  # Batch dot product
        # pos_scores = contract(
        #     "ij,ij->i", u_embs, self.v_embs(pos_v), backend="torch"
        # )
        pos_scores = F.logsigmoid(pos_scores)

        neg_scores = torch.bmm(n, c.unsqueeze(2)).squeeze()
        neg_scores.register_hook(lambda x: x.clamp(min=-10, max=10))
        # neg_scores = torch.einsum(
        #     "ijk,ikl->ijl", [self.v_embs(neg_v), u_embs.unsqueeze(2)]
        # )  # Batch matrix multiplication
        # neg_scores = contract(
        #     "ijk,ikl->ijl",
        #     self.v_embs(neg_v),
        #     u_embs.unsqueeze(2),
        #     backend="torch",
        # )  # Batch matrix multiplication
        neg_scores = F.logsigmoid(-1 * neg_scores).sum(dim=1)

        return torch.sum(-1 * (pos_scores + neg_scores))


class CBOW(Word2Vec):
    def __init__(self, emb_size, emb_dimension, cbow_mean=True):
        super(CBOW, self).__init__(emb_size, emb_dimension)
        self.cbow_mean = cbow_mean

    def forward(self, target, context, negatives):
        t = self.syn1(target)
        c = self.syn0(context)
        n = self.syn1(negatives)

        # Mean of context vector without considering padding idx (0)
        if self.cbow_mean:
            mean_v_embs = torch.true_divide(
                c.sum(dim=1), (context != 0).sum(dim=1, keepdim=True),
            )
        else:
            mean_v_embs = c.sum(dim=1)

        # pos_scores = (u_embs * mean_v_embs).sum(dim=1)
        pos_scores = torch.mul(t, mean_v_embs)
        pos_scores.register_hook(lambda x: x.clamp(min=-10, max=10))
        pos_scores = torch.sum(pos_scores, dim=1)
        # pos_scores = torch.einsum(
        #     "ij,ij->i", [u_embs, mean_v_embs]
        # )  # Batch dot product
        # pos_scores = contract(
        #     "ij,ij->i",
        #     u_embs,
        #     mean_v_embs,
        #     backend="torch",
        #     optimize="dp",
        #     use_blas=True,
        # )
        pos_scores = F.logsigmoid(pos_scores)

        neg_scores = torch.bmm(n, mean_v_embs.unsqueeze(2)).squeeze()
        neg_scores.register_hook(lambda x: x.clamp(min=-10, max=10))
        # neg_scores = torch.einsum(
        #     "ijk,ikl->ijl", [self.v_embs(neg_v), u_embs.unsqueeze(2)]
        # )  # Batch matrix multiplication
        # neg_scores = contract(
        #     "ijk,ikl->ijl",
        #     self.v_embs(neg_v),
        #     u_embs.unsqueeze(2),
        #     backend="torch",
        #     optimize="dp",
        #     use_blas=True,
        # )  # Batch matrix multiplication
        neg_scores = F.logsigmoid(-1 * neg_scores).sum(dim=1)

        return torch.sum(-1 * (pos_scores + neg_scores))
