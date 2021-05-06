# Sememe-Based NLP Attack

This code is obtained from [source codes](https://github.com/thunlp/SememePSO-Attack) of "Word-level Textual Adversarial Attacking as Combinatorial Optimization".

## Requirements

- python == 3.7
- tensorflow-gpu == 1.14.0
- torch == 1.7.1
- keras == 2.2.4   
- sklearn == 0.0  
- anytree == 2.6.0  
- nltk == 3.4.5  
- OpenHowNet == 0.0.1a8    
- pytorch_transformers == 1.0.0  
- loguru == 0.3.2

## Required Data

The data we used is exactly the same as the one shared on [cloud](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/?p=%2F) by the author of "Word-level Textual Adversarial Attacking as Combinatorial Optimization" .


### Downloads:

- [IMDB all data](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/files/?p=%2FIMDB_used_data.zip)
- [SNLI all data](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/files/?p=%2FSNLI_used_data.zip)
- [SST-2 all data](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/files/?p=%2FSST_used_data.zip)

## Start Attack

Before attacking, you should edit the ``run*.py``, correct the data path.

For example, if we want to run **ls** attack on **IMDB** dataset, while the victim model is **BERT**.

```shell script
# currently on the project root
cd ./PSO_related/imdb
python run_ls_BERT.py
```

## Check Result

All results is saved in ``project_root/out/pso_related/{dataset}_{model}_{attack}/{SEED}/``

### Detailed results

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

Result folder has ``log.txt``, which shows the **Attack Success Rate, Query Number** and **Modification Rate**


#### Adversarial Text

First, run `generate_txt_results.py` with corresponding result data path.

The `.txt` version of adversarial examples is generated and stored in `adv.txt`, the corresponding original sentences is in `orig.txt`


