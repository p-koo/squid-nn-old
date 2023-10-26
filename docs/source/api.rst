.. _api:

API Reference
==============

Examples
--------

The script ``run_squid.py`` can be modified and run as a demo for using the
SQUID framework in conjunction with two previously-published DNNs (DeepSTARR
and ResidualBind-32) to model a genomic locus using additive and pairwise effects,
respectively. 


Models
------

The ``squid.mutagenizer`` class contains functions to generate an *in silico*
dataset by randomly mutated an input sequence-of-interest.

.. autoclass:: squid.mutagenizer
    :members: apply_mut_by_seq_index
