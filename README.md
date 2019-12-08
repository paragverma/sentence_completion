# CSC 791: Natural Language Processing Project - Sentence Completion and Gap Filling Tasks

### Environment setup 
Exports of `conda env export` and `pip freeze` are inlcuded in the `\environment` directory as `environment.yml` and `requirements.txt` respectively.
To install required modules using pip `>pip install -r requirements.txt`, or
set up new conda environment.txt `>conda env create -f environment.yml`

#### Important Notes
Keep the directory structure same as this repository
Execute each file from the directory the file is in.
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


3. `test3.py':
Strategy: Create sentence vector from word vectors for each sentence by ignoring the blank. Obtain word vectors for each option. Compute cosine distance between each option and sentence, least distance option is chosen. Word Vectors obtained using BERT.

    See instructions for starting BERT server in `test2.py`. Start BERT server with some aggregation strategy, edit the file with IP address of server.
    
    Run `> python test3.py`

4. `test4.py`:
Strategy: For each option, replace blank with the option, then using BERT, calculate total cross entropy loss. The option which causes least loss is the chosen option.

    Run `> python test4.py`

5. `test5.py`:
Strategy: Predict blank using BERT, the calculate word2vec(Corpus trained) similarity between predicted word and options to chose closest option. Also for Fine-tune BERT as well.

    - In `test5.py`, edit this line `w2v_model = gensim.models.Word2Vec.load(os.path.join(os.getcwd(), os.pardir, "word2vec_models", "word2vec_model_skip_window12.model"))`, with `word2vec_model_skip_window4.model`, `word2vec_model_skip_window8.model` or `word2vec_model_skip_window12.model` for different window sizes, or your own word2vec model.
    
    - For using Fine-tuned BERT model, or your own BERT model, comment lines `14-16`, i.e. 
        ```
        bertmodelname = 'bert-large-uncased-whole-word-masking'
        tokenizer = BertTokenizer.from_pretrained(bertmodelname)
        model = BertForMaskedLM.from_pretrained(bertmodelname)
        ```
        and uncomment lines `20-24`, i.e.
        ```
        bertmodelname = 'bert-large-uncased-whole-word-masking'
        tokenizer = BertTokenizer.from_pretrained(bertmodelname)
        bertsavedmodelname = "pytorch_model3.bin"
        model_state_dict = torch.load(bertsavedmodelname)
        model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=bertmodelname, state_dict=model_state_dict)
        ```
        
        Replace `"pytorch_model3.bin"` with `"/bert_models/pytorch_model1.bin"` for first 100,000 sentences, `"/bert_models/pytorch_model1.bin"` for first 200,000 sentences, `"/bert_models/pytorch_model3.bin"` for first 300,000 sentences, or your own BERT model. 
    
    - (Optional) For fine-tuning BERT, run `run_lm_finetuning.py`. This script is taken from https://github.com/huggingface/transformers. Usage:
        ```
        python run_lm_finetuning.py \
        --output_dir=output \
        --model_type=bert \
        --model_name_or_path=bert-large-uncased-whole-word-masking \
        --do_train \
        --train_data_file=$TRAIN_FILE \
        ```
        `$TRAIN_FILE` should be `/cleaned_corpus/cleaned_data.txt`
    - Run `> python test5.py`

6. `test6.py`:
Strategy: Same as `test2.py`, predict blank using BERT, the calculate BERT vectors similarity between predicted word and options to chose closest option. In this experiment, we perform the aggregations ourselves instead of pooling functions of the library.

    Steps:
        - Start BERT-server
            - `> cd bert_models`
            - Start bert-server (see instructions for `test2.py`), with pooling strategy set to NONE, i.e. pass `--pooling_strategy NONE` while starting server
            - `> python test6.py`
