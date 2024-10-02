# transformer-from-scratch
Implementing a Transformer from scratch in PyTorch. Assignment 2 of Advanced NLP (IIIT-H Monsoon '24). 
___
## Dependencies
### General
The dependencies can be imported in conda using the envs file in `docs`. 
```sh
conda env create -f docs/envs.yml
```

Alternatively, the specific dependencies (obtained using `conda env export --from-history`) are:
- nltk
- numpy
- pytorch
- pytorch-cuda
- torchaudio (probably redundant)
- torchvision (probably redundant)
- python (<=3.11)
- spacy

### Specific

Following the above, install the english and french tokenizers using
```sh
python -m spacy download en
python -m spacy download fr
```
___
## Usage
## Cleaning / Processing
The data can be cleaned by calling the following function
```sh
python -m src.clean
```
This outputs a new, processed corpus in a separate directory. 

### Training
The train module takes in 4 hyper-params, the `dropout rate`, `embedding dimension`, `number of layers` in the encoder and decoder of the model, and `number of heads` in the multi-headed attention. 

An example train command could be:
```sh
python -m src.train --dropout 0.3 --embed_dim 240 --num_layers 3 --num_heads 2
```
The model will be saved in the format `transformer_en_fr-<dropout>-<emb-dim>-<num-layers>-<num-heads>.pth`. 

### Testing
The test module takes in the path of the pre-trained model to test. It outputs a BLEU score in the file res, with the name `translations_with_bleu-<dropout>-<emb-dim>-<num-layers>-<num-heads>.txt`. 

An example test command could be:
```sh
python -m src.test "transformer_en_fr-0.3-240-3-2.pth"
```
___
## Pre-trained models
Some pre-trained models on various values of the above hyper-params can be found in [this](https://drive.google.com/drive/folders/1b8qnuLzz-PF50hJpnwir4tJ5jSFRlZSX?usp=sharing) google drive link. 

Their corresponding results with loss variation and bleu score files can also be found there. 
The `final` directories denote experimental models whose parameters were determined on tuning. They are expected to have better results than the rest. 

The current best BLEU achieved with my limited compute is 0.2021. 
___
## TODO
- add more hyper-parameters as arguments
- add corpus as argument

___
___
