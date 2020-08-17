import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Word2Vec(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(Word2Vec, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension

        # u_embs: embedding for target word
        # v_embs: embedding for context words
        self.u_embs = nn.Embedding(emb_size, emb_dimension, sparse=True, padding_idx=0)
        self.v_embs = nn.Embedding(emb_size, emb_dimension, sparse=True, padding_idx=0)

        init_range = 0.5 / self.emb_dimension
        init.uniform_(self.u_embs.weight.data, -init_range, init_range)
        init.constant_(self.v_embs.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        raise NotImplementedError

    def save_embeddings(self, id2word, output_vec_path: str, vec_format="txt"):
        assert vec_format in ["txt", "pkl"]
        if not os.path.exists(os.path.dirname(output_vec_path)):
            os.makedirs(os.path.dirname(output_vec_path))
        embs = self.u_embs.weight.cpu().data.numpy()
        output_vec_path = os.path.splitext(output_vec_path)[0]
        if vec_format is None or vec_format == "txt":
            if not os.path.exists(output_vec_path + ".txt"):
                print("Save embeddings to " + output_vec_path + ".txt")
                with open(output_vec_path + ".txt", "w") as f:
                    f.write("%d %d\n" % (len(id2word), self.emb_dimension))
                    for wid, w in id2word.items():
                        e = " ".join(map(lambda x: str(x), embs[wid]))
                        f.write("%s %s\n" % (w, e))
            else:
                raise FileExistsError("'" + output_vec_path + ".txt' already exists")
        else:
            if not os.path.exists(output_vec_path + ".pkl"):
                print("Save embeddings to " + output_vec_path + ".pkl")
                embs_tmp = {w: embs[wid] for wid, w in id2word.items()}
                pickle.dump(embs_tmp, open(output_vec_path + ".pkl", "wb"))
            else:
                raise FileExistsError("'" + output_vec_path + ".pkl' already exists")
        print("Done")


class SkipGram(Word2Vec):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGram, self).__init__(emb_size, emb_dimension)

    def forward(self, pos_u, pos_v, neg_v):
        u_embs = self.u_embs(pos_u)
        v_embs = self.v_embs(pos_v)

        score = torch.mul(u_embs, v_embs)
        score = torch.sum(score, dim=1)
        # score = torch.einsum("ij,ij->i", [u_embs, v_embs])  # Batch dot product
        score = F.logsigmoid(score)

        neg_v_embs = self.v_embs(neg_v)
        neg_score = torch.bmm(neg_v_embs, u_embs.unsqueeze(2))
        # neg_score = torch.einsum(
        #     "ijk,ikl->ijl", [neg_v_embs, u_embs.unsqueeze(2)]
        # )  # Batch matrix multiplication
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1 * (torch.sum(score) + torch.sum(neg_score))


class CBOW(Word2Vec):
    def __init__(self, emb_size, emb_dimension, cbow_mean=True):
        super(CBOW, self).__init__(emb_size, emb_dimension)
        self.cbow_mean = cbow_mean

    def forward(self, pos_u, pos_v, neg_v):
        u_embs = self.u_embs(pos_u)
        v_embs = self.v_embs(pos_v)

        # Mean of context vector without considering padding idx (0)
        if self.cbow_mean:
            mean_v_embs = torch.true_divide(
                torch.sum(v_embs, dim=1), (pos_v != 0).sum(dim=1).unsqueeze(dim=1)
            )
        else:
            mean_v_embs = torch.sum(v_embs, dim=1)

        score = torch.mul(u_embs, mean_v_embs)
        score = torch.sum(score, dim=1)
        # score = torch.einsum("ij,ij->i", [u_embs, torch.mean(v_embs, 1)])  # Batch dot product
        score = F.logsigmoid(score)

        neg_v_embs = self.v_embs(neg_v)
        neg_score = torch.bmm(neg_v_embs, u_embs.unsqueeze(2))
        # neg_score = torch.einsum(
        #     "ijk,ikl->ijl", [neg_v_embs, u_embs.unsqueeze(2)]
        # )  # Batch matrix multiplication
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1 * (torch.sum(score) + torch.sum(neg_score))
