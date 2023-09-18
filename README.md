# README

<br/><br/>

![logo_dark](./images/logo_dark.png#gh-dark-mode-only)
![logo_light](./images/logo_light.png#gh-light-mode-only)

<br/><br/>

## SQUID Repository
This repository contains the software implementation for our [paper](https://www.google.com) **Title** (Seitz, Kinney* and Koo*). It contains tools to apply the discussed method **SQUID** (**S**urrogate **Qu**antitative **I**nterpretability for **D**eepnets) on genomic models. This work was developed in the Kinney and Koo research groups at Cold Spring Harbor Laboratory.

## Install:

```bash
pip install squid
```

Dependencies:

```bash
conda create -n squid python=3
pip install mavenn
pip install mavenn --upgrade
pip install pyyaml
```

Note: for older versions of Tensorflow, ... #to be done


### Usage:
SQUID provides a simple interface that takes as input deep-learning models. For any deep-learning model that takes in sequence as input, perform SQUID as follows:

```python
import squid

#to be done
```

The `run_squid.py` script contains code for running SQUID on several example deep-learning models.

## Examples on Google Colab:

- DeepSTARR analysis: https://

- ResidualBind-32 analysis: https://


## Attribution:
If this code is useful in your work, please cite:

### License:
Copyright (C) 2022–2023 Evan Seitz, Peter Koo, Justin Kinney

The software, code sample and their documentation made available on this website could include technical or other mistakes, inaccuracies or typographical errors. We may make changes to the software or documentation made available on its web site at any time without prior notice. We assume no responsibility for errors or omissions in the software or documentation available from its web site. For further details, please see the LICENSE file.
