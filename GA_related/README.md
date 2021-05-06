# Embedding/LM-Based NLP Attack

This code is obtained from [source codes](https://github.com/nesl/nlp_adversarial_examples) of papaer "Generating Natural Language Adversarial Examples".

## Requirements

- python == 3.7
- tensorflow-gpu == 1.14.0
- torch == 1.7.1
- keras == 2.2.4  

## Prepare Data

### IMDB Dataset

1) Download the [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) and unzip

2) Download the [glove vector embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip) (used by the model)

3) Download the [counter-fitted vectors](https://raw.githubusercontent.com/nmrksic/counter-fitting/master/word_vectors/counter-fitted-vectors.txt.zip) (used by our attack)

4) Build the vocabulary and embeddings matrix.
```
python build_embeddings.py
```

That will take like a minute, and it will tokenize the dataset and save it to a pickle file. It will also compute some auxiliary files like the matrix of the vector embeddings for words in our dictionary. All files will be saved under `./GA/aux_files` directory created by this script.

5) Train the sentiment analysis model.
```
python train_model.py
```

Our model training result: test accuracy =  88.55%


6) Download the Google language model.
```
./download_googlm.sh
```

7) Pre-compute the distances between embeddings of different words (required to do the attack) and save the distance matrix.

```
python compute_dist_mat.py 
```

### SNLI Dataset

1) Download the [dataset using](https://nlp.stanford.edu/projects/snli/snli_1.0.zip)

2) Download the [Glove](http://nlp.stanford.edu/data/glove.840B.300d.zip) and [Counter-fitted Glove embedding vectors](https://raw.githubusercontent.com/nmrksic/counter-fitting/master/word_vectors/counter-fitted-vectors.txt.zip)

3) Train the NLI model
```
python snli_rnn_train.py
```

Our model training result: test accuracy =  82.31%


4) Pre-compute the embedding matrix 
```
python nli_compute_dist_matrix.py
```




## Start Attack

```shell script
# Attack on IMDB
python run_ls_imdb.py
python run_ga_imdb.py

# Attack on SNLI
python run_ls_snli.py
python run_ga_snli.py
```

## Check Results

All results is saved in ``project_root/out/GA_related/{dataset}_{model}_{attack}/{SEED}/``

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


### Attack Performance

Result folder has ``log.txt``.  The **Attack Success Rate, Query Number** and **Modification Rate** is presented at the end of the  ``log.txt`` file

### Adversarial Text

First, run `generate_txt_results_{dataset}.py` with corresponding result data path.

The `.txt` version of adversarial examples is generated and stored in `adv.txt`, the corresponding original sentences is in `orig.txt`

