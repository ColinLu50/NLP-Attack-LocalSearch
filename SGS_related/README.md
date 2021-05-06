# Morpholog-based NLP Attack

This code is obtained from [source codes](https://github.com/salesforce/morpheus) of "It's Morphin' Time! Combating Linguistic Discrimination with Inflectional Perturbations".

## Requirements

* python==3.7.9
* torch==1.7.1
* transformers==4.0.1
* allennlp==0.9.0
* fairseq==1.0.0
* lemminflect==0.2.1
* nltk==3.5
* sacremoses==0.0.43

## Process Data

### Dataset

* `SQuAD 2.0`: Download official version [here](https://rajpurkar.github.io/SQuAD-explorer/). We only attak on development set, put `dev-v2.0.json` under  `SGS_related/processed_dataset/` directory. 
* `newstest2014 En-Fr`: Download official version [here](http://matrix.statmt.org/test_sets/newstest2014.tgz?1504722373)

For `newstest2014 En-Fr` dataset, you should first unzip the ``newstest2014.tgz`` and run ``process_sgm.py`` with correct filepath. 

Or you can directly use the processed plain text version under `SGS_related/processed_dataset/` directory.

### Model

For BERT-based models on `SQuAD 2.0`, we adopt the transformer implementation of [BERT](https://huggingface.co/twmkn9/albert-base-v2-squad2) and [SpanBERT](https://huggingface.co/mrm8488/spanbert-large-finetuned-squadv2). 
These two model will be automatically downloaded when you first start attack.

For BiDAF models, we use allennlp implementation of [GloVe-BiDAF]("https://allennlp.s3.amazonaws.com/models/bidaf-model-2017.09.15-charpad.tar.gz") and [ELMo-BiDAF]("https://s3.us-west-2.amazonaws.com/allennlp/models/bidaf-elmo-model-2018.11.30-charpad.tar.gz").
These two model will be automatically downloaded when you first start attack.

For NMT task, we use fairseq implementation of [Convolutional Sequence to Sequence model](https://github.com/pytorch/fairseq/blob/master/examples/conv_seq2seq/README.md) and [Transformer-Big model](https://github.com/pytorch/fairseq/blob/master/examples/scaling_nmt/README.md). You should **manually** download this two model files, and configure the path in the `shared_params.py`.


### Start Attack

### Command

* Local Search

```
cd SGS_related/LS
python run_ls_{task}_{MODEL_TYPE}.py -m {model arg}
```

* Sequential Greedy Search

```
cd SGS_related/Morpheus
python run_morph_{task}_{MODEL_TYPE}.py -m {model arg}
```

### Args

* `-m` : model 
  - Value: bert, spanbert for `run_ls_qa_BERT.py`,  `run_morpheus_qa_BERT.py`
  - Value: bidaf, elmobidaf for `run_ls_qa_BiDAF.py`, `run_morpheus_qa_BiDAF.py`
  - Value: convS2S, transformer for `run_ls_nmt.py`, `run_morpheus_nmt.py`
  
### Examples

* Local Search Attack Example:

```
python run_ls_qa_BERT.py -m bert
``` 

* Morpheus SGS Attack Example:
```
python run_morpheus_nmt.py -m convS2S
```

## Check Results

The result folder is ``project_root/out/SGS_related/{model_name}_{search_method}/``

### Attack Performance

#### QA Task

Result folder includes ``answerable_logs.txt``. 
At the end of the log file, there is **Attack Success Rate, Query Number** and **Modification Rate** on **answerable** questions of SQuAD v2 .

Result folder may have ``all_logs.txt``. 
At the end of the log file, there is **Attack Success Rate, Query Number** and **Modification Rate** on **all** questions.


#### NMT Task

Result folder includes ``logs.txt``. 
At the end of the log file, there is **Attack Success Rate, Query Number** and **Modification Rate**  .



### Detailed Results

#### QA Task

A `.json` file which has exactly the same format as the original `dev-v2.0.json` , while the question is replaced by corresponding adversarial question.


#### NMT Task

Result folder includes a  ``.pkl`` file, whose element includes:

- is_attack_success : True if attack success
- src : orignal sentence
- ref : orignal translated sentence
- adv : adversarial sentence
- adv_pred: transalted outputs of adversarial sentence
- modif_rate: modification rate
  
