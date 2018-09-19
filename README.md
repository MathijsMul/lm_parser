## Introduction

Transition-based dependency parser using the arc hybrid system. Features are the hidden states of a pre-trained language model.
The language model used for testing is the best-performing English LSTM trained by [Gulordava e.a. (2018)](https://arxiv.org/abs/1803.1113), some of whose code was re-used here.
The parser is a single-layered MLP, as used in the final stage of [Kiperwasser & Goldberg (2016)](https://aclweb.org/anthology/Q16-1023).

## Requirements

The model was implemented in Python 3 using PyTorch. See the [website](https://pytorch.org/) to install the right version.

## Quickstart

From the command line `main.py` can be run to train a new model. E.g.:

    # On TRAIN_FILE, for 5 epochs, train an MLP with 64 hidden units, pre-trained language model stored at LM and training directory TRAIN_DIR
    python3 main.py --train $TRAIN_FILE --epochs 5 --hidden_units_mlp 64 --language_model $LM --train_directory $TRAIN_DIR --model_name 'example_model'

The script `example.sh` further illustrates the usage of `main.py`.

## TODO

- How to initialize hidden state of pre-trained LM? Currently just zeros. Use some kind of average?
- ROOT currently mapped to UNK
- Tune hyperparameters of MLP

