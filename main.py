import logging

from word2vec.word2vec import Word2Vec

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

if __name__ == "__main__":
    w2v = Word2Vec(
        train_file="./word2vec/data/dataset/text8",
        sentences_path="./sentences/sentences_text8.pkl",
        input_vocab_path=None,
        output_vocab_path="./vocab/vocab_text8.pkl",
        output_vec_path="./vec/vec_text8_5ws_5mc_10ns_100dim_3ep_cw",
        output_vec_format="txt",
        sg=0,
        window_size=5,
        min_count=5,
        ns_size=10,
        emb_dimension=100,
        epochs=3,
        lr_type="decay",
        num_workers=0,
    )
    w2v.train()
