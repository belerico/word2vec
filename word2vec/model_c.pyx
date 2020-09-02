import os
import pickle
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp

cdef np.float32_t[1000] EXP_TABLE

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CBOW:

    cdef int emb_size, emb_dimension
    cdef bint cbow_mean
    cdef np.float32_t[:, ::1] syn0, syn1
    cdef np.float32_t[::1] c_mean, e

    def __cinit__(self, int emb_size, int emb_dimension, bint cbow_mean=True):
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.cbow_mean = cbow_mean
        
        cdef np.float32_t init_range = 0.5 / emb_dimension
        self.syn0 = np.random.uniform(
            low=-init_range, high=init_range, size=(emb_size, emb_dimension)).astype('f')
        self.syn1 = np.zeros((emb_size, emb_dimension), dtype='f')
        cdef long j
        for j in range(emb_dimension):
            self.syn0[0, j] = 0
        self.e = np.zeros(emb_dimension, dtype='f')
        self.c_mean = np.zeros(emb_dimension, dtype='f')
        for i in range(1000):
            EXP_TABLE[i] = <np.float32_t>exp((i / <np.float32_t>1000 * 2 - 1) * 6)
            EXP_TABLE[i] = <np.float32_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))

    def forward(self, list targets, list context, list negatives, np.float32_t lr):
        self.c_forward(targets, context, negatives, lr)

    cdef c_forward(self, list targets, list context, list negatives, np.float32_t lr):
        cdef bint label
        cdef long i, j, d, wid_t, wid_c, idx_target, cw, c
        cdef np.float32_t score = 0, g = 0

        for i in range(len(targets)):
            target = targets[i]
            cw = 0

            for j in range(self.emb_dimension):
                self.e[j] = 0
                self.c_mean[j] = 0

            for c in context[i]:
                cw += 1
                for j in range(self.emb_dimension):
                    self.c_mean[j] += self.syn0[c, j]
            if cw != 0:
                for j in range(self.emb_dimension):
                    self.c_mean[j] /= cw

                for d in range(len(negatives[i]) + 1):
                    if d == 0:
                        idx_target = target
                        label = 1
                    else:
                        idx_target = negatives[i][d-1]
                        label = 0

                    if idx_target != 0:
                        score = 0
                        for j in range(self.emb_dimension):
                            score += self.syn1[idx_target, j] * self.c_mean[j]

                        if score <= -6 or score >= 6:
                            score = score >= 6
                        else:
                            score = EXP_TABLE[<int>((score + 6) * (1000 / 6 / 2))]
                        g = (label - score) * lr

                        for j in range(self.emb_dimension):
                            self.e[j] += g * self.syn1[idx_target, j]
                        for j in range(self.emb_dimension):
                            self.syn1[idx_target, j] += g * self.c_mean[j]

                for c in context[i]:
                    for j in range(self.emb_dimension):
                        self.syn0[c, j] += self.e[j]

    def save_embeddings(
        self, id2word, output_vec_path: str, vec_format="txt", overwrite=True
    ):
        assert vec_format in ["txt", "pkl"]
        if not os.path.exists(os.path.dirname(output_vec_path)):
            os.makedirs(os.path.dirname(output_vec_path))
        embs = self.syn0
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
                embs_tmp = {w: np.asarray(embs[wid]) for wid, w in id2word.items()}
                pickle.dump(
                    embs_tmp, open(output_vec_path + ".pkl", "wb"),
                )
            else:
                raise FileExistsError(
                    "'" + output_vec_path + ".pkl' already exists"
                )
        print("Done")
