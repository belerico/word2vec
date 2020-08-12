import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os


# u_embedding: Embedding for target word.
# v_embedding: Embedding for context words.


class SkipGram(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGram, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embs = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embs = nn.Embedding(emb_size, emb_dimension, sparse=True)

        init_range = 0.5 / self.emb_dimension
        init.uniform_(self.u_embs.weight.data, -init_range, init_range)
        init.constant_(self.v_embs.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        u_embs = self.u_embs(pos_u)
        v_embs = self.v_embs(pos_v)

        score = torch.mul(u_embs, v_embs).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)

        neg_v_embs = self.v_embs(neg_v)
        neg_score = torch.bmm(neg_v_embs, u_embs.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1 * (torch.sum(score) + torch.sum(neg_score))

    def save_embeddings(self, id2word, output_vec_path):
        if not os.path.exists(output_vec_path):
            print("Save embeddings to " + output_vec_path)
            if not os.path.exists(os.path.dirname(output_vec_path)):
                os.makedirs(os.path.dirname(output_vec_path))
            embs = self.u_embs.weight.cpu().data.numpy()
            with open(output_vec_path, "w") as f:
                f.write("%d %d\n" % (len(id2word), self.emb_dimension))
                for wid, w in id2word.items():
                    e = " ".join(map(lambda x: str(x), embs[wid]))
                    f.write("%s\t%s\n" % (w, e))
            print("Done")
        else:
            raise FileExistsError("'" + output_vec_path + "' already exists")
