# MolGAN
Pytorch implementation of MolGAN: An implicit generative model for small molecular graphs (https://arxiv.org/abs/1805.11973)  
This library refers to the following two source code.
* [nicola-decao/MolGAN](https://github.com/nicola-decao/MolGAN)
* [yunjey/StarGAN](https://github.com/yunjey/StarGAN)

## Dependencies

* **python>=3.5**
* **pytroch>=0.4.1**: https://pytorch.org
* **rdkit**: https://www.rdkit.org
* **numpy**

## Structure
* [data](https://github.com/yongqyu/MolGAN-pytroch/tree/master/data): should contain your datasets. If you run `download_dataset.sh` the script will download the dataset used for the paper (then you should run `data/sparse_molecular_dataset.py` to conver the dataset in a graph format used by MolGAN models).
* [models](https://github.com/yongqyu/MolGAN-pytorch/blob/master/models.py): Class for Models.

## Usage
```
python main.py
```

## Citation
```
[1] De Cao, N., and Kipf, T. (2018).MolGAN: An implicit generative
model for small molecular graphs. ICML 2018 workshop on Theoretical
Foundations and Applications of Deep Generative Models.
```

BibTeX format:
```
@article{de2018molgan,
  title={{MolGAN: An implicit generative model for small
  molecular graphs}},
  author={De Cao, Nicola and Kipf, Thomas},
  journal={ICML 2018 workshop on Theoretical Foundations
  and Applications of Deep Generative Models},
  year={2018}
}

```

Work In Progress.
