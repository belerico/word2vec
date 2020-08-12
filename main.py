import pickle

from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader

from word2vec.data.dataset import Word2vecDataset
from word2vec.data.vocab import Vocab
from word2vec.word2vec import Word2Vec

if __name__ == "__main__":
    # data = Vocab(train_file="./word2vec/data/dataset/sample.txt", min_count=5,)
    # data.init_vocab()
    # data.init_unigram_table()
    # data.init_keep_table()

    # print("Word freqs: ", "\n", data.word_freqs)
    # print("Word to IDs: ", "\n", data.word2id)
    # print("IDs to word: ", "\n", data.id2word)
    # print("Unigram table: ", "\n", data.unigram_table)
    # print("Sorted word IDs by freq: ", "\n", data.sorted)

    # dataset = Word2vecDataset(data, window_size=5)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     collate_fn=dataset.collate,
    # )

    # for i, batch in enumerate(dataloader):
    #     print("Batch number " + str(i))
    #     print("Target: ")
    #     print(batch[0])
    #     print("Context: ")
    #     print(batch[1])
    #     print("Negative: ")
    #     print(batch[2])

    w2v = Word2Vec(
        train_file="./word2vec/data/dataset/text8.txt",
        input_vocab_path=None,
        output_vocab_path="./vocab/vocab_text8.pkl",
        output_vec_path="./vec/vec_text8",
        output_vec_format="pkl",
        min_count=5,
        batch_size=1,
        emb_dimension=100,
        epochs=20,
        ns_size=10,
    )
    w2v.train()
    embs = pickle.load(open("./vec/vec_text8.pkl", "rb"))
    print("Cosine between 'cat' and 'dog' ", cosine(embs["dog"], embs["cat"]))
    print(
        "Cosine between 'cat' and 'anarchy' ", cosine(embs["cat"], embs["anarchy"]),
    )
    print(
        "Cosine between 'queen' and 'king' ", cosine(embs["queen"], embs["king"]),
    )
    print(
        "Cosine between 'king - man + woman' and 'queen' ",
        cosine(embs["queen"], embs["king"] - embs["man"] + embs["woman"]),
    )
