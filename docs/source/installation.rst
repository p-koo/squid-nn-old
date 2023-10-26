.. _installation:

Installation Instructions
=========================

using the ``pip`` package manager by executing the following at the
commandline: ::

    $ pip install squid

MAVE-based surrogate models and visualization tools used in our analysis require additional dependencies: ::

    $ pip install logomaker 
    $ pip install mavenn
    $ pip install mavenn --upgrade

Please note that the latest version of MAVE-NN conflicts with NumPy's version (1.24.3)
due to a conflicting dependency with TensorFlow. This is the reason
some users may have to run a pip upgrade command (shown above) 
after installation on their command line. 
Please see `MAVENN <https://mavenn.readthedocs.io>`_ for more information.

Alternatively, you can clone SQUID from
`GitHub <https://github.com/evanseitz/squid-nn>`_ 
using the command line: ::

    $ cd appropriate_directory
    $ git clone https://github.com/evanseitz/squid-nn.git

where ``appropriate_directory`` is the absolute path to where you would like
SQUID to reside. Then add the following to the top of any Python file in
which you use SQUID: ::

    # Insert local path to SQUID at beginning of Python's path
    import sys
    sys.path.insert(0, 'appropriate_directory/squid')

    #Load squid
    import squid


