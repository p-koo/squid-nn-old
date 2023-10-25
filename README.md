SQUID: interpreting sequence-based deep learning models for regulatory genomics
========================================================================
<!--- [![PyPI version](https://badge.fury.io/py/mavenn.svg)](https://badge.fury.io/py/mavenn) --->
<!--- [![Downloads](https://static.pepy.tech/personalized-badge/mavenn?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/mavenn) --->
<!--- [![Documentation Status](https://readthedocs.org/projects/mavenn/badge/?version=latest)](https://mavenn.readthedocs.io/en/latest/?badge=latest) --->

<br/>

![logo_dark](./images/logo_dark.png#gh-dark-mode-only)
![logo_light](./images/logo_light.png#gh-light-mode-only)

<br/>

**SQUID** (**S**urrogate **Qu**antitative **I**nterpretability for **D**eepnets) is a TensorFlow package to interpret sequence-based deep learning models for regulatory genomics data with domain-specific surrogate models. For installation instructions, tutorials, and documentation, please refer to the SQUID website, https://squid-nn.readthedocs.io/. For an extended discussion of this approach and its applications, please refer to our manuscript:

* Seitz E., McCandlish D., Kinney J. and Koo P. Identifying *cis*-regulatory mechanisms from genomic deep neural networks using surrogate models.
<!--- <em>Genome Biol</em> **23**, 98 (2022). https://doi.org/10.1186/s13059-022-02661-7 --->
---

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

Note: For older DNNs that require inference via Tensorflow 1, Python < 3 is required which is not supported by MAVE-NN. Users will need to create separate environments in this case:
1. An environment with Tensorflow 1 and Python 2 for generating *in silico* MAVE data
2. An environment with Tensorflow 2 and Python 3 for training MAVE-NN surrogate models


### Usage:
SQUID provides a simple interface that takes as input a sequence-based deep-learning model (e.g., a DNN), which is used as an oracle to generate an *in silico* MAVE dataset representing a localized region of sequence space. The MAVE dataset can then be fit using a domain-specific surrogate model, with the resulting parameters visualized to reveal the biological mechanisms driving model performance.

<img src="images/framework.png" alt="fig" width="1000"/>

- Examples in Python: 

	- The script `run_squid.py` contains code for running SQUID on several example deep-learning models
	- The `squid_manuscript` repository contains examples for applying SQUID on several other DNNs such as ENFORMER, and includes additional options for scalability: https://github.com/evanseitz/squid_manuscript

- Examples in Google Colab:

	- Additive analysis with DeepSTARR: https://colab.research.google.com/drive/12HR8Vu_8ji3Ac1wli4wgqx1J0YB73JF_?usp=sharing

	- Pairwise analysis with ResidualBind-32: https://colab.research.google.com/drive/1eKC78YE2l49mQFOlnA9Xr1Y9IO121Va5?usp=sharing


### Citation:
If this code is useful in your work, please our paper.

```bibtex
@article{seitz2023squid,
  title={Identifying *cis*-regulatory mechanisms from genomic deep neural networks using surrogate models},
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
