import pickle
import logging

from scipy.spatial.distance import cosine

from word2vec.data.dataset import Word2vecDataset
from word2vec.data.vocab import Vocab
from word2vec.word2vec import Word2Vec

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

if __name__ == "__main__":
    # data = Vocab.load_vocab("./vocab/vocab_sample.pkl")

    # data.init_vocab()
    # data.init_unigram_table()
    # data.init_keep_table()

    # print("Word freqs: ", "\n", data.word_freqs)
    # print("Word to IDs: ", "\n", data.word2id)
    # print("IDs to word: ", "\n", data.id2word)
    # print("Unigram table: ", "\n", data.unigram_table)
    # print("Sorted word IDs by freq: ", "\n", data.sorted)

    # dataset = Word2vecDataset(
    #     data, "./sentences/sentences_sample.pkl", window_size=5, sg=1
    # )

    # for epoch in range(5):
    #     for i, _ in enumerate(dataset):
    #         print(epoch, i)

    w2v = Word2Vec(
        train_file="./word2vec/data/dataset/sample.txt",
        input_vocab_path="./vocab/vocab_sample.pkl",
        output_vocab_path=None,
        output_vec_path="./vec/vec_sample_sg",
        output_vec_format="pkl",
        sg=1,
        min_count=5,
        batch_size=1,
        emb_dimension=10,
        epochs=100,
        ns_size=5,
        lr_type="decay",
        mikolov_context=True,
        num_workers=0,
    )
    w2v.train()
    embs = pickle.load(open("./vec/vec_sample_sg.pkl", "rb"))
    print(
        "Cosine between 'river' and 'flows' ",
        cosine(embs["river"], embs["flows"]),
    )
    print(
        "Cosine between 'river' and 'far' ", cosine(embs["river"], embs["far"])
    )
