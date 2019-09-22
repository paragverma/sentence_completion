import preprocess
import pickle
from gensim.models import Word2Vec
import gensim

def save_sentences():
    all_sentences, unique_tokens = preprocess.process_all_docs(directory="data/Holmes_Training_Data",
                                                               save_cleaned_docs=False,
                                                               break_at_sentence=10)
    with open("all_sentences.pkl", "wb") as f:
        pickle.dump(all_sentences, f)
    with open("unique_tokens.pkl", "wb") as f:
        pickle.dump(unique_tokens, f)

def load_sentences():
    all_sentences = []
    unique_tokens = None
    
    with open("all_sentences.pkl", "rb") as f:
        all_sentences = pickle.load(f)
    with open("unique_tokens.pkl", "rb") as f:
        unique_tokens = pickle.load(f)
    
    return all_sentences, unique_tokens

save_sentences()
all_sentences, unique_tokens = load_sentences()

model = Word2Vec(all_sentences, workers=8, size=300)
model.wv.save_word2vec_format("word2vec_model.gsm")

load_model = gensim.models.KeyedVectors.load_word2vec_format("word2vec_model_custom.gsm")