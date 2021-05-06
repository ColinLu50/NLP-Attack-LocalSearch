# WordNet/NE-based NLP Attack

This code is obtained from [source codes](https://github.com/JHL-HUST/PWWS) of "Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency".

## Requirements

### Python Packages
- python == 3.7.1
- tensorflow-gpu == 1.15.0
- spacy==2.1.4
- keras == 2.2.4   
- attrs == 18.2.0
- attr == 0.3.1
- scikit_learn == 0.21.2
- nltk == 3.4.5  

### Others

Download WordNet(a lexical database for the English language) by running ``nltk.download('wordnet')`` in python script.

Download spacy tokenizer in command line: ``python -m spacy download en_core_web_md`` and ``python -m spacy download en``


## Prepare Data

1) Download dataset files from [google drive](https://drive.google.com/open?id=1YdndNH0RE6BEpg04HtK6VWemYrowWzvA) , which include:
- IMDB: `aclImdb.zip`. Decompression and place the folder `aclImdb` in `SbGS_related/data_set/`.
- AG's News: `ag_news_csv.zip`. Decompression and place the folder `ag_news_csv` in `SbGS_related/data_set`.

2) Download `glove.6B.100d.txt` from [google drive](https://drive.google.com/open?id=1YdndNH0RE6BEpg04HtK6VWemYrowWzvA) and place the file in `SbGS_related`, or you can customize the `GLOVE_REAL_PATH` in `neural_networks.py`.

3) Run `training.py` or use command like `python training.py --model word_cnn --dataset imdb --level word`. You can reset the model hyper-parameters in `neural_networks.py` and `config.py`.

<!-- `PWWS_related/runs/`contains used pretrained NN models, the performance of these models are showed as the following table: -->

| data_set       | neural_network | test_set | 
| -------------- | -------------- | -------- | 
| IMDB           | word_cnn       | 88.26%  | 
|                | word_bdlstm    | 85.71%  | 
| AG's News      | word_cnn       | 90.76%  | 
|                | char_cnn       | 89.36%  | 

<!-- If you want to use these model, rename the them or modify the paths to model in the corresponding `.py` files. -->


## Start Attack

### Command

```
python run_{search method}.py -m {model} -d {dataset} -l {level}
```

### Args

- model: word_cnn, word_bdlstm, char_cnn
- datset: imdb, agnews 
- level: word, char
  
### Examples

* SbGS Attack Example:

```command
cd ./SbGS_related/PWWS
python run_PWWS.py --model word_cnn --dataset imdb --level word
```

* LS Attack Example:

```command
# SbGS attack
cd ./SbGS_related/LS
python run_LS.py --model word_cnn --dataset imdb --level word
```


## Check Results

The result folder is ``project_root/out/SbGS_related/{search method}_{dataset}/{model_name}``

### Attack Performance

Result folder has ``log.txt``.
The **Attack Success Rate, Query Number** and **Modification Rate** is presented at the end of the  ``log.txt`` file


### Detailed Results

Result folder incudes  `.pkl` type files:
* `test_list.pkl` : attack indices corresponding to the original test data.

* `success.pkl`: success attack information, tuple of three lists of same length.
  - success_idx_list: success attack indices, corresponding to the original test data.
  - success_examples_list: examples of success attack
  - success_y_list: the predictions of success attacking adversarial examples, different from the original label.

* `long.pkl`: actual change the model prediction but fail because of high modification rate, tuple of three lists of same length.
  - long_fail_idx_list: indices of examples that fail because of high modification, corresponding to the original test data.
  - success_examples_list: examples of failing examples because of high modification rate.
  - success_y_list: the predictions of adversarial examples, different from the original label.


### Adversarial Text Results

Run ``generate_txt_results.py`` to generate text of attack results, remember to set correct folder path.

The `.txt` version of adversarial examples is generated and stored in `adv.txt`, the corresponding original sentences is in `orig.txt`



