import re


mask_token = "[MASK]"


def clean_msr_sent_gen(cls_on=True, mask_on=True, mode="bert"):
    
    def func(string):
        global mask_token
        if(mode == "bert"):
            mask_token = "[MASK]"
        elif(mode == "roberta"):
            mask_token = "<mask>"
        else:
            raise Exception("mode not correct")
        
        string = clean_str_simple(string, mask_on=mask_on)
        if(cls_on):
            if(mode == "bert"):
                string = "[CLS] " + string
            elif(mode == "roberta"):
                string = "<s> " + string + " </s>"
        
        return string
    
    return func

def clean_msr_sentence(string, cls_on=True, mask_on=True):
    
    string = clean_str_simple(string, mask_on=mask_on)
    if(cls_on):
        string = "[CLS] " + string
    
    return string

multiple_whitespace_re = re.compile(r"\s+", flags=re.I)
apostrophe_re = re.compile(r"(\w+)'s", flags=re.I)
non_text_re = re.compile(r"[^\w ']")

def clean_str_simple(string, mask_on=True):
    string = string.strip().lower()
    if(mask_on):
        string = re.sub(r"_{3,}", mask_token, string)
    string = re.sub(apostrophe_re, r"\1", string)
    string = re.sub(non_text_re, " ", string)
    string = string.replace("_", "")
    string = re.sub(multiple_whitespace_re, " ", string)
    string = string.replace(mask_token[1:-1], mask_token)
    return string

def clean_str_simple_mod(string):
    string = string.strip().lower()

    string = re.sub(apostrophe_re, r"\1", string)
    string = re.sub(non_text_re, " ", string)
    string = string.replace("_", "")
    string = re.sub(multiple_whitespace_re, " ", string)

    string = re.sub(r"(^|\s)(')($|\s)", r"\1\3", string)
    string = re.sub(r"'\s", r"", string)
    string = re.sub(r"(\S+)'\s", r"\1", string)
    
    string = re.sub(r"(\s)'(\w+)(\s)", r"\1\2\3", string)
    return string.strip()

import os
import unidecode
import spacy
import logging

logging.info("Loading Spacy")
#spacy_nlp = spacy.load("en_core_web_lg")
#spacy_nlp.max_length *= 4
logging.info("Loaded")

import nltk
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

def process_all_docs(directory="../data/Holmes_Training_Data", save_cleaned_docs=True, break_at_sentence=None):
    count_files = 0

    #count_disclaimers = 0
    max_len = 0
    #all_lines_raw = []
    #print("punkt2")
    #n = 3
    if(not os.path.isdir(os.path.join(os.getcwd(), "cleaned_data_bert"))):
        os.mkdir(os.path.join(os.getcwd(), "cleaned_data_bert"))
    
    if(save_cleaned_docs):
        wp = open(os.path.join(os.getcwd(), "cleaned_data_bert.txt"), "w+")
                
    temp_i = 0
    for root, dirs, files in os.walk(directory, topdown=True): 
        for file in files:
            #n -= 1
            #if(n < 0):
            #    wp.close()
            #    return
            temp_i += 1
            if(not file.lower().endswith("txt")):
                continue
            count_files += 1
            print(file)
            
            with open(os.path.join(os.getcwd(), root, file), "r", encoding="latin1") as rp:
                
                #spacy_parse = spacy_nlp(unidecode.unidecode(rp.read()))
                filecontent = unidecode.unidecode(rp.read())
                search_disclaimer_end = re.search("(END).*(THE)\s*(SMALL)\s*?(PRINT).*", filecontent, flags=re.I)
                
                filecontent = filecontent[search_disclaimer_end.span()[1]:]
                
                """spacy_parse = spacy_nlp(filecontent)
                
                
                for sent in spacy_parse.sents:
                    line = sent.text
                    #print(line)
                    #all_lines_raw.append(line)
                    clean_sentence = clean_str_simple_orig(line)
    
                    if(save_cleaned_docs):
                        if(len(clean_sentence.strip()) > 1):
                            wp.write(clean_sentence.strip() + "\n")"""
                
                sentences = tokenizer.tokenize(filecontent)
                
                
                for sent in sentences:
                    line = sent
                    clean_sentence = clean_str_simple_mod(line)
                    max_len = max(max_len, len(clean_sentence))
                    if(save_cleaned_docs):
                        if(len(clean_sentence.strip()) > 1):
                            wp.write(clean_sentence.strip() + "\n")

            if(save_cleaned_docs):
                wp.write("\n")
                
    
            
            if(break_at_sentence):
                if(temp_i == break_at_sentence):
                    break
                
    if(save_cleaned_docs):
        wp.close()
            
            
    print("maxlen", max_len)
    return
            