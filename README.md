# README

<br/><br/>

![logo_dark](./images/logo_dark.png#gh-dark-mode-only)
![logo_light](./images/logo_light.png#gh-light-mode-only)

<br/><br/>

## SQUID Repository
This repository contains the software implementation for our [paper](https://www.google.com) **Title** (Seitz, McCandlish, Kinney and Koo). It contains tools to apply the discussed method **SQUID** (**S**urrogate **Qu**antitative **I**nterpretability for **D**eepnets) on genomic models.

SQUID is a TensorFlow package to interpret sequence-based deep learning models for regulatory genomics data with domain-specific surrogate models.

For questions, email: koo@cshl.edu

<img src="images/schematic.png" alt="fig" width="1000"/>


### Install:

```bash
pip install squid-nn
```

### Dependencies:

```bash
conda create -n squid python=3
pip install logomaker
pip install mavenn
pip install mavenn --upgrade
```

Note: For older DNNs that require inference via Tensorflow 1, Python < 3 is required which is currently unsupported by MAVE-NN. Users will need to create separate environments in this case: one using TF1/Python2 for generating in silico MAVE data; and another using TF2/Python3 for training MAVE-NN surrogate models.


### Usage:
SQUID provides a simple interface that takes as input a deep-learning model that takes in sequence as input.

- Examples in Python: 

	- `run_squid.py` contains code for running SQUID on several example deep-learning models.

- Examples in Google Colab:

	- Additive analysis with DeepSTARR: https://colab.research.google.com/drive/12HR8Vu_8ji3Ac1wli4wgqx1J0YB73JF_?usp=sharing

	- Pairwise analysis with ResidualBind-32: https://colab.research.google.com/drive/1eKC78YE2l49mQFOlnA9Xr1Y9IO121Va5?usp=sharing


### Citation:
If this code is useful in your work, please our paper.

```bibtex
@article{seitz2023squid,
  title={TBD},
  author={Seitz, Evan and McCandlish, David and Kinney, Justin and Koo, Peter},
  journal={TBD},
  volume={TBD},
  number={TBD},
  pages={TBD},
  year={2023},
  publisher={TBD}
}
```

### License:
Copyright (C) 2022–2023 Evan Seitz, David McCandlish, Justin Kinney, Peter Koo

The software, code sample and their documentation made available on this website could include technical or other mistakes, inaccuracies or typographical errors. We may make changes to the software or documentation made available on its web site at any time without prior notice. We assume no responsibility for errors or omissions in the software or documentation available from its web site. For further details, please see the LICENSE file.
