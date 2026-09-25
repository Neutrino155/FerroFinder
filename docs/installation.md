# Installation

FerroFinder requires Python 3.10 or newer. Install the package in a virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The install includes ASE, NumPy, NetworkX, SciPy, and spglib. The interactive
explorer uses browser-native HTML and JavaScript. Matplotlib is only needed to
regenerate the 250-material screening plots:

```bash
python -m pip install -e '.[plots]'
```

## MACE-Field

Install a MACE-Field version compatible with the checkpoint you plan to use.
FerroFinder does not download models or install MACE-Field automatically.
The checkpoint must provide the field-aware head named by `--head` (the
default is `mp-ferroelectric`).

If the correct MACE-Field source is not the one Python imports by default,
point FerroFinder at its source checkout:

```bash
ferrofinder polar.cif \
  --model /path/to/MACEField.model \
  --mace-checkout /path/to/mace-field
```

The equivalent environment setting is `FERROFINDER_MACE_CHECKOUT`. An
explicit `--mace-checkout` takes precedence.

For supported structure formats and a complete first run, see the
[user guide](user-guide.md).
