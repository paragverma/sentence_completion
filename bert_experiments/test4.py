#BERT sentence loss
#Substitute [MASK] with each option and calculate loss. Min loss wins

import pandas as pd
import bert
import bert_preprocess

df = pd.read_csv("../data/testing_data.csv")

answers_df = pd.read_csv("../data/test_answer.csv")

ground_truth = list(answers_df["answer"].values)

df["question"] = df["question"].apply(bert_preprocess.clean_msr_sent_gen(cls_on=False, mask_on=True))


#df["predicted"] = df["question"].apply(bert.predict_missing)


ground_truth = list(answers_df["answer"].values)


options = ['a)','b)','c)','d)','e)']
preds = []
ix_to_option_dict = {i+1:option for i, option in enumerate(options)}
for i, row in df.iterrows():
    sentence = row["question"]
    preds_i = []
    print(i)
    for option in options:
        mod_sent = sentence.replace("[MASK]", row[option])
        
        preds_i.append(bert.sentence_loss(mod_sent))
    
    #print(preds_i)
    preds.append(ix_to_option_dict[preds_i.index(min(preds_i)) + 1][:-1])


from sklearn.metrics import accuracy_score
print(accuracy_score(ground_truth, preds))

#0.25384615384615383