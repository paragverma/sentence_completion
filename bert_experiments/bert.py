import torch
from transformers import BertForMaskedLM, BertTokenizer
from torch.nn import CrossEntropyLoss
import math
from sentence_transformers import SentenceTransformer
from transformers import XLNetModel, XLNetTokenizer
import bert_preprocess
import numpy as np

st_model = SentenceTransformer('bert-base-nli-mean-tokens')



bertmodelname = 'bert-large-uncased-whole-word-masking'
tokenizer = BertTokenizer.from_pretrained(bertmodelname)
model = BertForMaskedLM.from_pretrained(bertmodelname)


"""
bertmodelname = 'bert-large-uncased-whole-word-masking'
tokenizer = BertTokenizer.from_pretrained(bertmodelname)
bertsavedmodelname = "pytorch_model3.bin"
model_state_dict = torch.load(bertsavedmodelname)
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=bertmodelname, state_dict=model_state_dict)
"""


#text = '[CLS] I want to [MASK] the car because it is cheap . [SEP]'

def predict_missing(text):
    
    
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    
    #indexed_tokens = tokenizer.encode(text)
    
    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    # Load pre-trained model (weights)
    model.eval()
    
    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    
    masked_index = tokenized_text.index(bert_preprocess.mask_token)
    
    
    predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    
    if("roberta" in bertmodelname or 'xlnet' in bertmodelname):
        predicted_token = tokenizer.convert_tokens_to_string(predicted_token).strip()

    return predicted_token

def sentence_loss(text):
    
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    
    #indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens])
    model.eval()
    with torch.no_grad():
        predictions = model(tokens_tensor)[0]
    
    ce_loss = CrossEntropyLoss()
    loss = ce_loss(predictions.squeeze(),tokens_tensor.squeeze()).data 
    return math.exp(loss)

def st_model_sentvec_list(textlist):
    return st_model.encode(textlist)


def sentence_to_vec(vectors, strategy="average"):
    ret_vec = np.zeros(vectors.shape)
    
    veclen = 0
    if(strategy == "average"):
        for i in range(vectors.shape[0]):
            if(np.count_nonzero(vectors[i]) > 0):
                ret_vec += vectors[i]
                veclen += 1
            else:
                break
        
        return ret_vec / vectors.shape[0]
    
    if(strategy == "ema"):
        veclen = 0
        for i in range(vectors.shape[0]):
            if(np.count_nonzero(vectors[i]) > 0):
                veclen += 1
            else:
                break
        
        alpha = 2.0 / (veclen + 1)
        
        for i in range(veclen):
            ret_vec = (ret_vec * (1 - alpha)) + (vectors[i] * alpha)
        
        return ret_vec
    