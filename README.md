# RNN on wikitext2
Pytorch of RNN and LSTM on wikitext2 dataset.
---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-orange)](https://pytorch.org/)


## What's This Project About?

This project implemented RNN and LSTM language models by using PyTorch default functions.
The model uses 30 time steps, a 100-dimensional embedding space, two hidden layers
using a *tanh()* activation function, tied embeddings and 20 batches during training.

In other words, the hyper-parameters are:

| **Hyper-parameter** |       **Value**       |
|:-------------------:|:---------------------:|
|      batch size     |           20          |
|   sequence length   |           30          |
|   number of layers  |           2           |
|    embedding size   |          100          |
|     hidden size     |          100          |
|    learning rate    | (tune by experiments) |

The LSTM model also applied drop-out and gradient clipping.

## Dataset

The data used are [wikitext2](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)
as presented in `./data` directory. The data files in the directory are preprocessed,
low frequency words replaced with `<unk>`, the sentences are tokenized.

## Structure of the Project

The structure of the project a]is as follows:

```bash
.
├── data
│   ├── wiki.test.txt
│   ├── wiki.train.txt
│   └── wiki.valid.txt
├── LICENSE
├── README.md
├── rnn_model.ipynb
└── src
    ├── data_module.py
    ├── data_prep.py
    ├── lstm_pl.py
    └── rnn_pl.py
```

A jupyter notebook version is in the root directory that can be uploaded along with
datas to Google colab or AWS sagemaker.

## Dependencies

The following dependencies are required, please install by using the virtual environment
of your choice:

```
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install pytorch_lightning
pip install nltk
python -m nltk.downloader punkt
pip install tensorboard
```

## How to Run?

To run the project, navigate to `./src` and use commend `python lstm_pl.py` or
`rnn_pl.py`. Alternatively, you can load the jupyter notebook.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.
