import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# u_embedding: Embedding for target word.
# v_embedding: Embedding for context words.


class SkipGram(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGram, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        init_range = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -init_range, init_range)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)

        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)

        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1 * (torch.sum(score) + torch.sum(neg_score))
