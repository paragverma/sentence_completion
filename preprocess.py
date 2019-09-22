import os, re
import unidecode
from collections import Counter
import spacy
import logging

logging.basicConfig(level=logging.INFO)

stopword_list = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

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
    string = re.sub(r"[^A-Za-z0-9'\-]", " ", string)
    string = re.sub(r"\b[0-9]+\b", "__NUMBER__", string)
    
    #string = re.sub(r"\b\'([\w'\-0-9])\'?", "\b\1\b", string)
    string = re.sub(r"\s\'([\w\'\-0-9]+[\w\-0-9])\'?", r" \1 ", string)
    string = re.sub(r"\s([\w\'\-0-9]+)\'[\s$]", r" \1 ", string)
    return string.strip().lower()



logging.info("Loading Spacy")
spacy_nlp = spacy.load("en_core_web_lg")
spacy_nlp.max_length *= 4
logging.info("Loaded")

def clean_token(string):
    if(string.startswith("'")):
        string = string[1:]
    if(string.endswith("'")):
        string = string[:-1]
    """if(ignore_stopwords):
        if(string in stopword_list):
            return """""
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
                    #print(line)
                    #all_lines_raw.append(line)
                    clean_sentence = clean_str(line)
                    clean_sentence = " ".join([token.lemma_ for token in spacy_nlp(clean_sentence) if token.lemma_ != "-PRON-"])
                    #print(clean_sentence)
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
    return " ".join([clean_token(token) for token in sentence.strip().split()])          
    #return " ".join(list(map(clean_token, sentence.strip().split())))            
                

