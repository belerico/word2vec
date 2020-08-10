from word2vec.data.data import InputData

if __name__ == "__main__":
    data = InputData("./word2vec/data/sample.txt")
    data.init_vocab()
    print(data.word_freqs)
    data.init_unigram_table()
    print(data.unigram_table)
    data.init_keep_table()
    print(data.keep_table)
