import os, re
import unidecode
from collections import Counter
import spacy
import logging

logging.basicConfig(level=logging.INFO)

def clean_str(string):
    string = string.replace("\n", " ")
    string = re.sub(r"\[([^\]]+)\]", " ", string)
    string = re.sub(r"\(([^\)]+)\)", " ", string)
    string = re.sub(r"[^A-Za-z0-9,!?.;\']", " ", string)
    string = re.sub(r"\b(\d+[\.\,]?)+[\.\,]?\b", "__NUMBER__", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"\s{2,}", " ", string)
    """string = re.sub(
        # there is a space or an end of a string after it
        r"[^\w#@&]+(?=\s|$)|"
        # there is a space or beginning of a string before it
        # not followed by a number
        r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
        # not in between numbers and not . or @ or & or - or #
        # e.g. 10'000.00 or blabla@gmail.com
        # and not url characters
        r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
        " ",
        string
    )"""
    string = re.sub(r"[^A-Za-z0-9'\-]", " ", string)
    string = re.sub(r"\b[0-9]+\b", "__NUMBER__", string)
    
    #string = re.sub(r"\b\'([\w'\-0-9])\'?", "\b\1\b", string)
    string = re.sub(r"\s\'([\w\'\-0-9]+[\w\-0-9])\'?", r" \1 ", string)
    string = re.sub(r"\s([\w\'\-0-9]+)\'[\s$]", r" \1 ", string)
           
    return string.strip().lower()



logging.info("Loading Spacy")
spacy_nlp = spacy.load("en_core_web_lg")
spacy_nlp.max_length *= 2
logging.info("Loaded")

def clean_token(string):
    if(string.startswith("'")):
        string = string[1:]
    if(string.endswith("'")):
        string = string[:-1]
    return string
    
def process_all_docs(directory="data/Holmes_Training_Data", save_cleaned_docs=True, break_at_sentence=None):
    count_files = 0
    count_sentences = 0
    count_tokens = 0
    #count_disclaimers = 0
    unique_tokens = Counter()
    all_sentences = []
    #all_lines_raw = []
    
    if(not os.path.isdir(os.path.join(os.getcwd(), "cleaned_data"))):
        os.mkdir(os.path.join(os.getcwd(), "cleaned_data"))
    
    temp_i = 0
    for root, dirs, files in os.walk(directory, topdown=True): 
        for file in files:
            temp_i += 1
            if(not file.lower().endswith("txt")):
                continue
            count_files += 1
            print(file)
            if(save_cleaned_docs):
                wp = open(os.path.join(os.getcwd(), "cleaned_data", file), "w+")
            with open(os.path.join(os.getcwd(), root, file), "r", encoding="latin1") as rp:
                
                #spacy_parse = spacy_nlp(unidecode.unidecode(rp.read()))
                filecontent = unidecode.unidecode(rp.read())
                search_disclaimer_end = re.search("(END).*(THE)\s*(SMALL)\s*?(PRINT).*", filecontent, flags=re.I)
                
                filecontent = filecontent[search_disclaimer_end.span()[1]:]
                
                spacy_parse = spacy_nlp(filecontent)
                
                for sent in spacy_parse.sents:
                    line = sent.text
                    #all_lines_raw.append(line)
                    clean_sentence = clean_str(line)
                    if(len(clean_sentence.split()) > 2):
                        count_sentences += 1
                        count_tokens += len(clean_sentence.split())
                        clean_sentence_tokenized = list(map(clean_token, clean_sentence.split()))
                        clean_sentence_tokenized = [token for token in clean_sentence_tokenized if len(token) > 0]
                        for token in clean_sentence_tokenized:
                            unique_tokens[token] += 1
                        all_sentences.append(clean_sentence_tokenized)
                    if(save_cleaned_docs):
                        wp.write(clean_sentence + "\n")
            
            if(save_cleaned_docs):
                wp.close()
            
            if(break_at_sentence):
                if(temp_i == break_at_sentence):
                    break
    
    return all_sentences, unique_tokens
            

def clean_sentence_raw(sentence):
    sentence = clean_str(sentence)
    return " ".join(list(map(clean_token, sentence.strip().split())))            
                

