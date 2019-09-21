import os, re
import unidecode

def clean_str(string):
    string = re.sub(r"\[([^\]]+)\]", " ", string)
    string = re.sub(r"\(([^\)]+)\)", " ", string)
    string = re.sub(r"[^A-Za-z0-9,!?.;]", " ", string)
    string = re.sub(r"\b(\d+[\.\,]?)+[\.\,]?\b", "__NUMBER__", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(
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
    )
    string = re.sub(r"\b[0-9]+\b", "__NUMBER__", string)
    return string.strip().lower()

if(not os.path.isdir(os.path.join(os.getcwd(), "cleaned_data"))):
    os.mkdir(os.path.join(os.getcwd(), "cleaned_data"))

count_files = 0
count_sentences = 0
count_tokens = 0
count_disclaimers = 0
for root, dirs, files in os.walk("data/Holmes_Training_Data", topdown=True): 
    for file in files:
        if(not file.lower().endswith("txt")):
            continue
        count_files += 1
        print(file)
        with open(os.path.join(os.getcwd(), "cleaned_data", file), "w+") as wp:
            with open(os.path.join(os.getcwd(), root, file), "r", encoding="latin1") as rp:
                startreading = False
                for line in unidecode.unidecode(rp.read()).split("\n"):
                    if(not startreading):
                        if(re.search("(END).*(THE)\s*(SMALL)\s*?(PRINT)", line, flags=re.I)):
                            startreading = True
                            #print(line)
                            count_disclaimers += 1
                        continue
                    clean_sentence = clean_str(line)
                    count_sentences += 1
                    count_tokens += len(clean_sentence.split())
                    wp.write(clean_sentence + "\n")
                if(not startreading):
                    print("{} file did not have a disclaimer".format(file))

                
                

