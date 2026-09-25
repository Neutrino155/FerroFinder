# Contributing

Changes should keep the endpoint-only input contract explicit: inference sees
the polar structure, not a known parent or path. Keep model-relative candidate
evidence separate from claims that require relaxation, electronic-structure,
or experimental validation.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

MACE-Field and checkpoint weights are not installed by this project. For a
real inference run, follow the [installation notes](docs/installation.md).

## Before submitting changes

- Keep CIF, XYZ, and extended XYZ input behavior documented.
- Preserve unit-cell and polarization conventions in any scientific change.
- Avoid committing checkpoints, generated run outputs, or machine-specific
  paths.
- Add focused tests for behavioral changes and run the relevant tests locally.

Open an issue for a large scientific or interface change before investing in
implementation. Include the physical assumption, units, a matched control,
and the evidence needed to support the proposed change.
