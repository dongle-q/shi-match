<img src="logo/logo.svg" alt="shi-match" width="400">

SHI-match is a Python tool that uses automatic differentiation, conjugate gradient, and singular value decomposition
to match beta-SUMO and alpha-SOMO for open-shell molecules.

# Installation
SHI-match requires these Python dependencies:

- JAX
- Numpy
- Scipy

It is highly recommended that you install these packages with pip::

```bash
pip install jax numpy scipy
```

# Usage
To use SHI-match, first you need to prepare file47 input, which contains SCF matrices.
See documentation at https://jautschbach.github.io/shi-match/ for more information.

Simply run :

```bash
shi-match.py <file_47>
```

The output will be saved in :file:`shi_match.log`.

To specify hardware you want to use, you can run with ``-hw`` or ``---hardware`` option

```bash
shi-match.py <file47> -hw <hardware>
```

Accpetable hardware options are: ``cpu``, ``cuda`` (NVIDIA GPU), ``rocm`` (AMD GPU) and ``tpu``. The default is ``cpu``.
