import gensim
import pandas as pd
import preprocess

w2v_model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)



def sentence_to_vec(sentence, strategy):
    pass

def similarity(vec1, vec2):
    pass

def predict_answers(df, q_col="question", options=['a)','b)','c)','d)','e)']):
    
    df = df[q_col].apply(preprocess.clean_sentence_raw)
    
    ix_to_option_dict = {i+1:option for i, option in enumerate(options)}
    
    preds = []
    for i, row in df.iterrows():
        sentence_vector = sentence_to_vec(row[q_col])
        preds_i = []
        for option in options:
            word_vector = w2v_model[row[option]]
            preds_i.append(similarity(sentence_vector, word_vector))
        
        preds.append(ix_to_option_dict[preds_i.index(max(preds_i))][:-1])
    return preds

df = pd.read_csv("data/testing_data.csv")
answers_df = pd.read_csv("data/test_answer.csv")


from sklearn.metrics import accuracy_score
predictions = predict_answers(df)
ground_truth = list(answers_df["answer"].values)

print("Accuracy:", accuracy_score(ground_truth, predictions))