Installation
=============

SHI-match requires these Python dependencies:

- JAX
- Numpy
- Scipy

It is highly recommended that you install these packages with pip::

    pip install jax numpy scipy

For around 500 orbitals, SHI-match runs within 20 seconds on CPU. Therefore, the performance is quite good.

SHI-match can be downloaded via git command::

    git clone https://github.com/jautschbach/shi-match.git

To update the code, simply run::

    git pull

Run the command from the repository root with::

    src/shi-match.py <file47>

Alternatively, add the ``src`` directory to ``PATH`` in your shell configuration file (e.g. ``~/.bashrc``)::

    export PATH=$PATH:/path/to/shi-match/src
