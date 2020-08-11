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

    def forward(self, target, context, neg):
        emb_u = self.u_embeddings(target)
        emb_v = self.v_embeddings(context)
        emb_neg_v = self.v_embeddings(neg)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return score + neg_score
