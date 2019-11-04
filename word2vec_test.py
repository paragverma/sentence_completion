import gensim
import pandas as pd
import preprocess
import numpy as np
from scipy import spatial
import SIF_embedding
import tensorflow_hub as hub
import tensorflow as tf
from tqdm import tqdm
import re

import logging

logging.basicConfig(level=logging.WARNING)

#w2v_model = None
w2v_model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)


embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3") 

VECTOR_DIM = 300

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)




def sentence_filter_gen(ignore_stopwords=False, lemmatize=False):
    igs=ignore_stopwords
    lmtz=lemmatize
    
    def apply_funct(sentence):
        if(igs):
            sentence = " ".join([token for token in sentence.strip().split() if token not in preprocess.stopword_list])
        if(lmtz):
            sentence = " ".join([token.lemma_ for token in preprocess.spacy_nlp(sentence)])
        
        return sentence
    
    return apply_funct
    
def sentence_filter(sentence, ignore_stopwords=False, lemmatize=False):
    if(ignore_stopwords):
        sentence = " ".join([token for token in sentence.strip().split() if token not in preprocess.stopword_list])
    if(lemmatize):
        sentence = " ".join([token.lemma_ for token in preprocess.spacy_nlp(sentence)])
    
    #sentence = sentence.lower()
    #sentence = re.sub(apostrophe_re, r"\1", sentence)

    
    return sentence

def sentence_to_vec(sentence, strategy="average", ignore_stopwords=False, lemmatize=False, alpha=None, pc=1):
    
    global w2v_model
    global VECTOR_DIM
    
    if(len(sentence.strip()) == 0):
        return np.zeros((1, VECTOR_DIM))
    
    #print(sentence)
    
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
        if(word_count == 0):
            return np.zeros((1, VECTOR_DIM))
        
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
        
        if(len(word_vector_stack) == 0):
            return np.zeros((1, VECTOR_DIM))
        word_vector_stack = np.vstack(word_vector_stack)
        word_weight_stack = np.vstack(word_weight_stack)

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
    
    return 1 - spatial.distance.cosine(vec1, vec2)
    
    
def predict_answers(odf, preprocess_data=True, q_col="question", options=['a)','b)','c)','d)','e)'], pc=1, strategy="average", encoder="w2v", lemmatize=False, ignore_stopwords=False):
    
    df = odf.copy(deep=True)
    
    if(preprocess_data):
        df[q_col] = df[q_col].apply(func=preprocess.clean_str_simple)
    
    
    ix_to_option_dict = {i+1:option for i, option in enumerate(options)}
    
    preds = []
    
    df[q_col] = df[q_col].apply(sentence_filter_gen(ignore_stopwords=ignore_stopwords,
                                                      lemmatize=lemmatize))
    if(encoder == "use"):
        with tf.Session() as session: 
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sentence_vectors = session.run(embed(df[q_col].values))
            option_dict = dict()
            for option in options:
                option_dict[option] = session.run(embed(df[option].values))
        
        with tqdm(total=df.shape[0]) as pbar:
            for i, row in df.iterrows():
                preds_i = []
                for option in options:
                    preds_i.append(similarity(sentence_vectors[i, :], option_dict[option][i, :], strategy=strategy))
                    #print(sentence_vectors[i, :], option_dict[option][i, :])
                preds.append(ix_to_option_dict[preds_i.index(max(preds_i)) + 1][:-1])
                pbar.update(1)
        return preds
    
    
    with tqdm(total=df.shape[0]) as pbar:
        if(strategy == "wmd"):
        
            for i, row in df.iterrows():
                #sentence_vector = sentence_to_vec(row[q_col], strategy=strategy, pc=pc, ignore_stopwords=ignore_stopwords, lemmatize=lemmatize)
                sentence = row[q_col]
                #print("sentence", i, row[q_col])
                preds_i = []
                
                for option in options:
                    row_option = preprocess.clean_str_simple(row[option])
                    preds_i.append(w2v_model.wv.wmdistance(sentence, row_option))
                
                #print(preds_i)
                preds.append(ix_to_option_dict[preds_i.index(min(preds_i)) + 1][:-1])
                pbar.update(1)
            
            return preds
        
        elif(strategy == "n_similarity"):
        
            for i, row in df.iterrows():
                #sentence_vector = sentence_to_vec(row[q_col], strategy=strategy, pc=pc, ignore_stopwords=ignore_stopwords, lemmatize=lemmatize)
                sentence = row[q_col].strip().split()
                sentence = [word for word in sentence if word in w2v_model]
                #print("sentence", i, row[q_col])
                preds_i = []
                
                for option in options:
                    row_option = preprocess.clean_str_simple(row[option]).strip().split()
                    row_option = [word for word in row_option if word in w2v_model]
                    
                    if(len(sentence) < 1 or len(row_option) < 1):
                        preds_i.append(0.0)
                        continue
                    preds_i.append(w2v_model.wv.n_similarity(sentence, row_option))
    
                preds.append(ix_to_option_dict[preds_i.index(max(preds_i)) + 1][:-1])
                pbar.update(1)
            
            return preds
        
        else:
            for i, row in df.iterrows():
                sentence_vector = sentence_to_vec(row[q_col], strategy=strategy, pc=pc, ignore_stopwords=ignore_stopwords, lemmatize=lemmatize)
                
                #print("sentence", i, row[q_col])
                preds_i = []
                
                for option in options:
                    row_option = row[option]
                    row_option = preprocess.clean_str_simple(row_option).strip().split()[-1]
                    if row_option not in w2v_model:
                        preds_i.append(0.0)
                        continue
                    word_vector = w2v_model[row_option]
                    preds_i.append(similarity(sentence_vector, word_vector))
                preds.append(ix_to_option_dict[preds_i.index(max(preds_i)) + 1][:-1])
                pbar.update(1)
            
            return preds

df = pd.read_csv("data/testing_data.csv")

answers_df = pd.read_csv("data/test_answer.csv")
ground_truth = list(answers_df["answer"].values)


from sklearn.metrics import accuracy_score


print("Stopwords Not Removed, Words not lemmatized, Gnews Vectors")
predictions = predict_answers(df, strategy="average")
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema")
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="wmd")
print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="n_similarity")
print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=0)
print("Accuracy (SIF) 0:", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=1)
print("Accuracy (SIF) 1:", accuracy_score(ground_truth, predictions))




print("Stopwords Not Removed, Words Lemmatized, Gnews Vectors")
predictions = predict_answers(df, strategy="average", lemmatize=True)
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema", lemmatize=True)
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="wmd", lemmatize=True)
print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="n_similarity", lemmatize=True)
print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=0, lemmatize=True)
print("Accuracy (SIF) 0:", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=1, lemmatize=True)
print("Accuracy (SIF) 1:", accuracy_score(ground_truth, predictions))




print("Stopwords Removed, No Lemmatization, Gnews Vectors")
predictions = predict_answers(df, strategy="average", ignore_stopwords=True)
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema", ignore_stopwords=True)
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="wmd", ignore_stopwords=True)
print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="n_similarity", ignore_stopwords=True)
print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=0, ignore_stopwords=True)
print("Accuracy (SIF) 0:", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=1, ignore_stopwords=True)
print("Accuracy (SIF) 1:", accuracy_score(ground_truth, predictions))




print("Stopwords Removed, Words Lemmatized, Gnews Vectors")
predictions = predict_answers(df, strategy="average", ignore_stopwords=True, lemmatize=True)
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema", ignore_stopwords=True, lemmatize=True)
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="wmd", ignore_stopwords=True, lemmatize=True)
print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="n_similarity", ignore_stopwords=True, lemmatize=True)
print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=0, ignore_stopwords=True, lemmatize=True)
print("Accuracy (SIF) 0:", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=1, ignore_stopwords=True, lemmatize=True)
print("Accuracy (SIF) 1:", accuracy_score(ground_truth, predictions))



##############################################################################################
print("Stopwords Not Removed, No lemmatization, Universal Sentence Encoder")
predictions = predict_answers(df, encoder="use")
print("Accuracy:", accuracy_score(ground_truth, predictions))


print("Stopwords Not Removed, lemmatization performed, Universal Sentence Encoder")
predictions = predict_answers(df, lemmatize=True, encoder="use")
print("Accuracy:", accuracy_score(ground_truth, predictions))


print("Stopwords Removed, No Lemmatization, Universal Sentence Encoder")
predictions = predict_answers(df, strategy="average", ignore_stopwords=True, encoder="use")
print("Accuracy:", accuracy_score(ground_truth, predictions))


print("Stopwords Removed, lemmatization performed, Universal Sentence Encoder")
predictions = predict_answers(df, strategy="average", ignore_stopwords=True, lemmatize=True, encoder="use")
print("Accuracy:", accuracy_score(ground_truth, predictions))


##############################################################################################

w2v_model = gensim.models.KeyedVectors.load_word2vec_format("word2vec_model_skip_window8.gsm")

print("Stopwords Not Removed, Words not lemmatized, Custom Vectors")
predictions = predict_answers(df, strategy="average")
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema")
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="wmd")
print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="n_similarity")
print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=0)
print("Accuracy (SIF) 0:", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=1)
print("Accuracy (SIF) 1:", accuracy_score(ground_truth, predictions))





print("Stopwords Not Removed, Words Lemmatized, Custom Vectors")
predictions = predict_answers(df, strategy="average", lemmatize=True)
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema", lemmatize=True)
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="wmd", lemmatize=True)
print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="n_similarity", lemmatize=True)
print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=0, lemmatize=True)
print("Accuracy (SIF) 0:", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=1, lemmatize=True)
print("Accuracy (SIF) 1:", accuracy_score(ground_truth, predictions))






print("Stopwords Removed, No Lemmatization, Custom Vectors")
predictions = predict_answers(df, strategy="average", ignore_stopwords=True)
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema", ignore_stopwords=True)
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="wmd", ignore_stopwords=True)
print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="n_similarity", ignore_stopwords=True)
print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=0, ignore_stopwords=True)
print("Accuracy (SIF) 0:", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=1, ignore_stopwords=True)
print("Accuracy (SIF) 1:", accuracy_score(ground_truth, predictions))







print("Stopwords Removed, Words Lemmatized, Custom Vectors")
predictions = predict_answers(df, strategy="average", ignore_stopwords=True, lemmatize=True)
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema", ignore_stopwords=True, lemmatize=True)
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="wmd", ignore_stopwords=True, lemmatize=True)
print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="n_similarity", ignore_stopwords=True, lemmatize=True)
print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=0, ignore_stopwords=True, lemmatize=True)
print("Accuracy (SIF) 0:", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="sif", pc=1, ignore_stopwords=True, lemmatize=True)
print("Accuracy (SIF) 1:", accuracy_score(ground_truth, predictions))

###################################################################################

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    n = 0
    
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        
        try:
            embedding = np.array([float(val) for val in splitLine[1:]])
        except ValueError:
            print(n, word)
            n += 1
            continue
        model[word] = embedding
        n += 1


    print("Done.",len(model)," words loaded!")
    return model

w2v_model = loadGloveModel("glove.840B.300d.txt")

print("Stopwords Not Removed, Words not lemmatized, Glove Vectors")
predictions = predict_answers(df, strategy="average")
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema")
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

#predictions = predict_answers(df, strategy="wmd")
#print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

#predictions = predict_answers(df, strategy="n_similarity")
#print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))




#tmp_file = get_tmpfile(os.path.join(os.getcwd(), '.glove_to_word2vec.txt'))


print("Stopwords Not Removed, Words Lemmatized, Glove Vectors")
predictions = predict_answers(df, strategy="average", lemmatize=True)
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema", lemmatize=True)
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

#predictions = predict_answers(df, strategy="wmd", lemmatize=True)
#print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

#predictions = predict_answers(df, strategy="n_similarity", lemmatize=True)
#print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))





print("Stopwords Removed, No Lemmatization, Glove Vectors")
predictions = predict_answers(df, strategy="average", ignore_stopwords=True)
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema", ignore_stopwords=True)
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))

#predictions = predict_answers(df, strategy="wmd", ignore_stopwords=True)
#print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

#predictions = predict_answers(df, strategy="n_similarity", ignore_stopwords=True)
#print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))






print("Stopwords Removed, Words Lemmatized, Glove Vectors")
predictions = predict_answers(df, strategy="average", ignore_stopwords=True, lemmatize=True)
print("Accuracy (Average):", accuracy_score(ground_truth, predictions))

predictions = predict_answers(df, strategy="ema", ignore_stopwords=True, lemmatize=True)
print("Accuracy (EMA):", accuracy_score(ground_truth, predictions))



#predictions = predict_answers(df, strategy="wmd", ignore_stopwords=True, lemmatize=True)
#print("Accuracy (WMD):", accuracy_score(ground_truth, predictions))

#predictions = predict_answers(df, strategy="n_similarity", ignore_stopwords=True, lemmatize=True)
#print("Accuracy (n_similarity):", accuracy_score(ground_truth, predictions))
