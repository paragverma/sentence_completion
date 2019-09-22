import gensim
import pandas as pd
import preprocess
import numpy as np
from scipy import spatial
import SIF_embedding

w2v_model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)


VECTOR_DIM = 300

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

    
def sentence_to_vec(sentence, strategy="average", ignore_stopwords=False, lemmatize=False, alpha=None, pc=1):

    if(ignore_stopwords):
        sentence = " ".join([token for token in sentence.strip().split() if token not in preprocess.stopword_list])
    if(lemmatize):
        sentence = " ".join([token.lemma_ for token in preprocess.spacy_nlp(sentence)])
    if(len(sentence.strip()) == 0):
        return np.zeros((1, VECTOR_DIM))
    
    if(strategy == "average"):
        total_weight = np.zeros((1, VECTOR_DIM))
        word_count = 0
        for token in sentence.split():
            if token in w2v_model:
                total_weight += w2v_model[token].reshape(1, -1)
                word_count += 1
        
        return total_weight / word_count
    
    elif(strategy == "ema"):
        total_weight = np.zeros((1, VECTOR_DIM))
        word_count = 0
        for token in sentence.split():
            if token in w2v_model:
                word_count += 1
        
        if(alpha is None):
            alpha = 2.0 / (word_count + 1)
        for token in sentence.split():
            if token in w2v_model:
                total_weight = (total_weight * (1 - alpha)) + (w2v_model[token].reshape(1, -1) * alpha)
        
        return total_weight
    
    elif(strategy == "sif"):
        word_vector_stack = []
        word_weight_stack = []
        for i, token in enumerate(sentence.split()):
            if token in w2v_model:
                word_vector_stack.append(w2v_model[token].reshape(1, VECTOR_DIM))
                word_rank_inverse = 3000000 - w2v_model.vocab[token].count
                word_weight_stack.append(word_rank_inverse)
        
        word_vector_stack = np.vstack(word_vector_stack)
        word_weight_stack = np.vstack(word_weight_stack)
    
        #raise ValueError
    #word_weight_stack = word_weight_stack / np.sum(word_weight_stack)
    
    
        return SIF_embedding.SIF_embedding(word_vector_stack, word_weight_stack, pc)

        
        """
        for each word:
            add to word vector stack
            add weight to weight stack
            sentence_to_vec("he was turn knife knob", strategy="sif") 
        """
        
        
                

def similarity(vec1, vec2, strategy="cosine"):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    
    if(strategy == "cosine"):
        return 1 - spatial.distance.cosine(vec1, vec2)
    
    
def predict_answers(df, q_col="question", options=['a)','b)','c)','d)','e)'], strategy="average", lemmatize=False, ignore_stopwords=False):
    
    df[q_col] = df[q_col].apply(func=preprocess.clean_sentence_raw)
    
    ix_to_option_dict = {i+1:option for i, option in enumerate(options)}
    
    preds = []
    for i, row in df.iterrows():
        sentence_vector = sentence_to_vec(row[q_col], strategy=strategy, ignore_stopwords=ignore_stopwords, lemmatize=lemmatize)
        #print("sentence", i, row[q_col])
        preds_i = []
        for option in options:
            if row[option] not in w2v_model:
                preds_i.append(0.0)
                continue
            word_vector = w2v_model[row[option]]
            preds_i.append(similarity(sentence_vector, word_vector))
        preds.append(ix_to_option_dict[preds_i.index(max(preds_i)) + 1][:-1])
    return preds

df = pd.read_csv("data/testing_data.csv")
answers_df = pd.read_csv("data/test_answer.csv")
ground_truth = list(answers_df["answer"].values)


from sklearn.metrics import accuracy_score


print("Stopwords Included, Gnews Vectors")
predictions = predict_answers(df, strategy="average")
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema")
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif")
print("Accuracy (SIF):", accuracy_score(ground_truth, predictions))



print("Stopwords Removed, Gnews Vectors")
predictions = predict_answers(df, strategy="average", ignore_stopwords=True)
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema", ignore_stopwords=True)
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", ignore_stopwords=True)
print("Accuracy (SIF):", accuracy_score(ground_truth, predictions))


