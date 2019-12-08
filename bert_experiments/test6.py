#BERT predict word
#Calculate vec similarity between word and options (BERT server vectors)
#Custom vector aggregation


import pandas as pd
import bert
import bert_preprocess
from bert_serving.client import BertClient
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
    print(i)
    preds_i = []
    
    pred_vector = bert.sentence_to_vec(pred_vectors[i], strategy="average")
    
    for j in range(len(options)):
        option_vector = bert.sentence_to_vec(option_vectors[j][i], strategy="average")
        preds_i.append(similarity(pred_vector, option_vector))
    
    
    #print(preds_i)
    preds.append(ix_to_option_dict[preds_i.index(max(preds_i)) + 1][:-1])


from sklearn.metrics import accuracy_score
print(accuracy_score(ground_truth, preds))

#-2 Average: 0.41634615384615387
#-2 ema: 0.3855769230769231
#-1 ena: 0.41442307692307695
#-1 average: 0.4230769230769231

#Last layers are more finetuned to specific tasks, hence last layer works well