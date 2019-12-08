#BERT server encode sentence
#BERT server encode all options
#Calculate vector similarity

from bert_serving.client import BertClient
import pandas as pd
import bert_preprocess

bc = BertClient(ip='54.202.107.215')




df = pd.read_csv("../data/testing_data.csv")

answers_df = pd.read_csv("../data/test_answer.csv")

ground_truth = list(answers_df["answer"].values)

df["question"] = df["question"].apply(bert_preprocess.clean_msr_sent_gen(cls_on=False, mask_on=False))


from scipy import spatial
import numpy as np

def similarity(vec1, vec2, strategy="cosine"):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    
    return 1 - spatial.distance.cosine(vec1, vec2)

VECTOR_DIM = 1024
options = ['a)','b)','c)','d)','e)']
preds = []
ix_to_option_dict = {i+1:option for i, option in enumerate(options)}

sentence_vectors = bc.encode(list(df["question"].values))
option_vectors = []

for option in options:
    option_vectors.append(bc.encode(list(df[option].values)))

for i, row in df.iterrows():
    preds_i = []
    
    for j in range(len(options)):
        preds_i.append(similarity(sentence_vectors[i], option_vectors[j][i]))
    
    #print(preds_i)
    preds.append(ix_to_option_dict[preds_i.index(max(preds_i)) + 1][:-1])


from sklearn.metrics import accuracy_score
print(accuracy_score(ground_truth, preds))


#0.23942307692307693