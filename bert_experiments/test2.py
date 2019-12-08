#BERT predict word
#Calculate vec similarity between word and options (BERT server vectors)


import pandas as pd
import bert
import bert_preprocess
from bert_serving.client import BertClient
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

bc = BertClient(ip='18.237.186.56')


df = pd.read_csv("../data/testing_data.csv")

answers_df = pd.read_csv("../data/test_answer.csv")

ground_truth = list(answers_df["answer"].values)

df["question"] = df["question"].apply(bert_preprocess.clean_msr_sentence)


df["predicted"] = df["question"].apply(bert.predict_missing)


ground_truth = list(answers_df["answer"].values)

from scipy import spatial

VECTOR_DIM = 300

def similarity(vec1, vec2, strategy="cosine"):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    
    return 1 - spatial.distance.cosine(vec1, vec2)

#w2v_model = gensim.models.KeyedVectors.load_word2vec_format("../GoogleNews-vectors-negative300.bin.gz", binary=True)

pred_vectors = bc.encode(list(df["predicted"].values))
#pred_vectors = bc.encode(list(df["question"].values))


option_vectors = []
options = ['a)','b)','c)','d)','e)']

for option in options:
    option_vectors.append(bc.encode(list(df[option].values)))
    
preds = []
ix_to_option_dict = {i+1:option for i, option in enumerate(options)}
for i, row in df.iterrows():
    
    preds_i = []
                
    for j in range(len(options)):
        preds_i.append(similarity(pred_vectors[i], option_vectors[j][i]))
    
    
    #print(preds_i)
    preds.append(ix_to_option_dict[preds_i.index(max(preds_i)) + 1][:-1])


from sklearn.metrics import accuracy_score
print(accuracy_score(ground_truth, preds))

#-2 (Default) 0.4201923076923077
#Pooling -4 3 2 1, REDUCE_MEAN: 0.4355769230769231
#Pooling -4 3 2 1, REDUCE_MAX: 0.4288461538461538
#Pooling -4 3 2 1, REDUCE_MEAN_MAX: 0.4403846153846154
#Pooling -1 0.4423076923076923
#-1 whole sentence 0.22884615384615384

#Last layers are more finetuned to specific tasks, hence last layer works well