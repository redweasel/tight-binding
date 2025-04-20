# Tight-Binding Calculations

The main problem, that is solved by this package is fitting bandstructures using asymmetric tight binding models. This can be done automated or be broken down to control every part of the calculation.
The codebase also includes the following features, accessible as python classes:
- Plotting functions for bandstructures (`kpaths`) and Fermi-surfaces (`fermi_surface`) based on `matplotlib` and `skimage`.
- 3 fitting codes with slightly different goals:
  1. `BandStructureModel` for the most general types of linear tight-binding models.
  2. `AsymTightBinding` for typical Fourier series based tight-binding models. (it has partial support for overlap matrices as well, however fitting with them doesn't work)
  3. `TightBindingModel` a work in progress model for fitting symmetric models (doesn't quite work sadly...)
- An interface for easy computations with Quantum Espresso and interpretation of the resulting data. (Quantum Espresso not included)
- Symmetrisation code of arbitrary point-groups both for tight-binding models and for general tensors.
- Precise density of states computation with many options for interpolation (`DensityOfStates`).
- Precise bandgap computation (part of `DensityOfStates`)
- Work in progress code for computing integrals over the bandstructure for metals.

## Installation

This is a python package for Python >=3.11 and can therefore be installed using pip. At the moment it's not uploaded to PyPi, because it has so many WIP areas. To work with it, we recommend a virtual environment \[(1)[https://docs.python.org/3/library/venv.html], (2)[https://python.land/virtual-environments/virtualenv]\]. On Linux this means running the following two commands before running pip:
```
python3 -m venv ./.venv
source .venv/bin/activate
```

Then it can be installed by running the following command:

```
pip install git+https://github.com/redweasel/tight-binding.git#egg=tight-binding-redweasel
```

To use Quantum Espresso, you will need to make sure it is installed as well and available in the PATH.

## Getting Started

It's best to start with some examples to see some of the possibilities.
For this, clone this repository and go into the `examples` directory. Try running `fit_copper.py`, `dos_copper.py`, `fermi_surface_copper.py` and `bulk_copper.py`. The output of the fit is provided, so the scripts can be exectued in arbitrary order to explore without breaking anything.

If you have Quantum Espresso installed, you can also try to reproduce the computation of the bandstructure of copper by following the `ExampleQE.ipynb` notebook.
