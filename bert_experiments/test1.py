#BERT predict word
#Calculate w2v similarity between word and options (word2vec gensim googlenews)

import pandas as pd
import bert
import bert_preprocess
import os 
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


df = pd.read_csv("../data/testing_data.csv")

answers_df = pd.read_csv("../data/test_answer.csv")

ground_truth = list(answers_df["answer"].values)

df["question"] = df["question"].apply(bert_preprocess.clean_msr_sentence)


df["predicted"] = df["question"].apply(bert.predict_missing)


ground_truth = list(answers_df["answer"].values)

import gensim
from scipy import spatial
import numpy as np

VECTOR_DIM = 300

def similarity(vec1, vec2, strategy="cosine"):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    
    return 1 - spatial.distance.cosine(vec1, vec2)

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.getcwd(), os.pardir, "word2vec_models", "GoogleNews-vectors-negative300.bin.gz"), binary=True)

options = ['a)','b)','c)','d)','e)']
preds = []
ix_to_option_dict = {i+1:option for i, option in enumerate(options)}
for i, row in df.iterrows():
    pred_word = row["predicted"]
    preds_i = []
                
    for option in options:
        op_word = row[option]
        op_vec = np.zeros((1, VECTOR_DIM)) if op_word not in w2v_model else w2v_model[op_word]
        pred_vec = np.zeros((1, VECTOR_DIM)) if pred_word not in w2v_model else w2v_model[pred_word]
        preds_i.append(similarity(pred_vec, op_vec))
    
    #print(preds_i)
    preds.append(ix_to_option_dict[preds_i.index(max(preds_i)) + 1][:-1])


from sklearn.metrics import accuracy_score
print(accuracy_score(ground_truth, preds))

#0.5394328