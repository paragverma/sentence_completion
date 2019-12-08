from gensim.models import Word2Vec
import gensim

all_sentences = []

with open("cleaned_corpus/cleaned_data.txt", "r") as f:
    for line in f.readlines():
        all_sentences.append(line.split())

model = Word2Vec(all_sentences, sg=1, window=12, workers=8, size=300)
model.save("word2vec_model_skip_window12.model")

load_model = gensim.models.KeyedVectors.load_word2vec_format("word2vec_model_custom.gsm")