# CSC 791: Natural Language Processing Project - Sentence Completion and Gap Filling Tasks

### Environment setup 
Exports of `conda env export` and `pip freeze` are inlcuded in the `\environment` directory as `environment.yml` and `requirements.txt` respectively.
To install required modules using pip `>pip install -r requirements.txt`, or
set up new conda environment.txt `>conda env create -f environment.yml`
### Directory Overview
- **baseline_and_refinements.py:** This file contains code for the Baseline Model and its Refinements
- **/bert_experiments:** This  directory contains experiments related to BERT. All experiments are in files named *test_<num>.py*. More details are given in later sections
- **/data:**
  - **testing_data.csv:** Contains 1040 sentences with blanks, and the options for each blank
  - **test_answer.csv:** Contains the answer to each of the 1040 sentences, i.e. the correct option for each blank
- **preprocess.py:** Module required for *baseline_and_refinements.py*. Contains preprocessing code for baseline model and its refinements
- **train_w2v.py:** Code to train own word2vec model for baseline and its refinements
- **cleaned_corpus:** Preprocessed and aggregated data into a single file from the 19C corpus. Download from here (https://drive.google.com/open?id=15rxIrt4YuWvsjzcKKAD9XQMieUTyMHVd) 
- **word2vec_models:** Pretrained Word2Vec models (including GoogleNews). Download from here (https://drive.google.com/open?id=15rxIrt4YuWvsjzcKKAD9XQMieUTyMHVd) 
- **bert_models:** Fine tuned BERT models on the corpus (Upto first 300,000 sentences). Download from here (https://drive.google.com/open?id=15rxIrt4YuWvsjzcKKAD9XQMieUTyMHVd)

### Experiments

#### Baseline, Refined Baseline
Make sure environment is set up (See Environment section)  
Download `/word2vec_models` directory  
Run:
`> python baseline_and_refinements.py`

Will print out results of Baseline + Refinements

#### BERT
Make sure environment is set up (See Environment section)
Download `/word2vec_models`, `/cleaned_corpus` and `\bert_models` directories.

##### Experiments
1. `test1.py`:  
Strategy: Predict blank using BERT, the calculate word2vec(Googlenews Pre-trained) similarity between predicted word and options to chose closest option.
`> python test1.py`: Prints out accuracy with this approach

2. `test2.py`:
Strategy: Predict blank using BERT, the calculate BERT vectors similarity between predicted word and options to chose closest option.

    Steps:
    - Start BERT-server
        - `> cd bert_models`
        - Depending on strategy, use appropriate command to start server. The BERT server can be started on a remote machine with appropriate computational power as well. Keep ports 5555 and 5556 open. `uncased_L-24_H-1024_A-16` is the BERT-large-uncases-whole-word-masking model, downloaded from Google's official BERT repository
        - `> bert-serving-start -model_dir uncased_L-24_H-1024_A-16` (Second last layer, Average Pool)
        - `> bert-serving-start -pooling_layer -4 -3 -2 -1 -model_dir uncased_L-24_H-1024_A-16` (Last 4 layers, Average Pool)
        - `> bert-serving-start -pooling_layer -1 -model_dir uncased_L-24_H-1024_A-16` (Last Layer, Average Pool)
        - `> bert-serving-start -pooling_strategy REDUCE_MAX -model_dir uncased_L-24_H-1024_A-16` (Second last layer, Max Pool)
        - `> bert-serving-start -pooling_layer -4 -3 -2 -1 -pooling_strategy REDUCE_MAX -model_dir uncased_L-24_H-1024_A-16` (Last 4 layers, Max Pool)
        
        - In general, `-pooling_layer` argument specifies layer to use. Can pass list of layers, prefixed with '-'. Valid `-pooling_strategy` arguments are REDUCE_MEAN, REDUCE_MAX, REDUCE_MEAN_MAX, NONE. For more comprehensive instructions, see https://github.com/hanxiao/bert-as-service
    
    - Edit the line `bc = BertClient(ip='18.237.186.56')`, to the IP address of the machine where BERT server is running
    - Run `> python test2.py`


