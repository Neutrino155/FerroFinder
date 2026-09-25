"""Small MACE-Field adapter for polarization and Born-charge responses."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .polarization import cartesian_to_reduced
from .structures import clean_copy, require_periodic


def resolve_mace_checkout(
    explicit: str | Path | None = None, *, prepare_import: bool = False
) -> Path:
    """Resolve MACE-Field from an explicit path, environment, or Python import."""

    configured = explicit or os.environ.get("FERROFINDER_MACE_CHECKOUT")
    if configured:
        checkout = Path(configured).expanduser().resolve()
        if not checkout.is_dir():
            raise ValueError(f"MACE checkout does not exist: {checkout}")
        if prepare_import:
            loaded = sys.modules.get("mace")
            loaded_file = getattr(loaded, "__file__", None) if loaded else None
            if loaded_file and checkout not in Path(loaded_file).resolve().parents:
                raise ImportError(
                    f"MACE is already loaded from {loaded_file}, outside {checkout}; "
                    "start a fresh Python process"
                )
            if str(checkout) not in sys.path:
                sys.path.insert(0, str(checkout))
                importlib.invalidate_caches()
        return checkout

    loaded = sys.modules.get("mace")
    origin = getattr(loaded, "__file__", None) if loaded else None
    if origin is None:
        spec = importlib.util.find_spec("mace")
        origin = spec.origin if spec is not None else None
        if origin is None and spec is not None and spec.submodule_search_locations:
            origin = next(iter(spec.submodule_search_locations), None)
    if origin is None:
        raise ImportError(
            "MACE-Field is not importable; install a compatible build or set "
            "--mace-checkout or FERROFINDER_MACE_CHECKOUT"
        )
    module_path = Path(origin).expanduser().resolve()
    for parent in (module_path.parent, *module_path.parents):
        if (parent / ".git").exists() and any(
            (candidate / "__init__.py").is_file()
            for candidate in (parent / "mace", parent / "src" / "mace")
        ):
            return parent
    return module_path.parent


@dataclass
class ModelEvaluation:
    """Energy, Berry polarization, and optional field-response tensors."""

    energy: float
    polarization: np.ndarray
    reduced_polarization: np.ndarray
    becs: np.ndarray
    polarizability: np.ndarray | None = None


def _array(results: dict, key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = results.get(key)
    if value is None:
        raise RuntimeError(
            f"MACE-Field did not return {key!r}; select a field-aware checkpoint head"
        )
    array = np.asarray(value, dtype=float)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"MACE result {key!r} has shape {array.shape}, expected {shape}")
    return array.reshape(shape)


class MACEFieldModel:
    """Load one field-aware checkpoint and evaluate endpoint recovery inputs."""

    def __init__(
        self,
        model: str | Path | Sequence[str | Path],
        *,
        head: str = "mp-ferroelectric",
        device: str = "cpu",
        dtype: str = "float64",
        mace_checkout: str | Path | None = None,
    ) -> None:
        self.model_paths = (
            [str(model)] if isinstance(model, (str, Path)) else [str(x) for x in model]
        )
        if not self.model_paths:
            raise ValueError("at least one model checkpoint is required")
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        self.head = head
        self.device = device
        self.dtype = dtype
        self.mace_checkout = mace_checkout
        self._calculator = None
        self._polarization_calculator = None
        self._property_calculator = None
        self.evaluation_count = 0
        self.full_response_evaluations = 0
        self.energy_polarization_evaluations = 0

    def make_calculator(self):
        """Load the selected MACE-Field calculator once."""

        if self._calculator is not None:
            return self._calculator
        previous_disable = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            checkout = resolve_mace_checkout(self.mace_checkout, prepare_import=True)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
                    category=UserWarning,
                )
                try:
                    import mace
                    from mace.calculators import MACECalculator
                except ImportError as exc:  # pragma: no cover - runtime dependency
                    raise ImportError(
                        "Install a compatible MACE-Field build or set "
                        "--mace-checkout to its source checkout"
                    ) from exc
                origin = Path(mace.__file__).resolve()
                if checkout != origin and checkout not in origin.parents:
                    raise ImportError(
                        f"MACE-Field was resolved at {checkout} but imported from {origin}"
                    )
                self._calculator = MACECalculator(
                    model_paths=self.model_paths,
                    model_type="MACEField",
                    head=self.head,
                    device=self.device,
                    default_dtype=self.dtype,
                    electric_field=None,
                    compute_polarization=True,
                    compute_becs=True,
                    compute_polarizability=False,
                )
        finally:
            logging.disable(previous_disable)
        available = getattr(self._calculator, "available_heads", None)
        if available is not None and self.head not in available:
            raise ValueError(
                f"head {self.head!r} is unavailable; choices are {list(available)!r}"
            )
        return self._calculator

    def make_polarization_calculator(self):
        """Reuse the loaded model without computing Born charges."""

        if self._polarization_calculator is not None:
            return self._polarization_calculator
        response_calculator = self.make_calculator()
        try:
            from mace.calculators import MACECalculator

            self._polarization_calculator = MACECalculator(
                models=response_calculator.models,
                model_type="MACEField",
                head=self.head,
                device=self.device,
                default_dtype=self.dtype,
                electric_field=None,
                compute_polarization=True,
                compute_becs=False,
                compute_polarizability=False,
            )
        except (ImportError, TypeError, ValueError, AttributeError) as exc:
            raise RuntimeError(
                "The selected MACE-Field build cannot provide polarization-only "
                "evaluations"
            ) from exc
        return self._polarization_calculator

    def make_property_calculator(self):
        """Reuse the loaded checkpoint with BEC and polarizability outputs on."""

        if self._property_calculator is not None:
            return self._property_calculator
        response_calculator = self.make_calculator()
        try:
            from mace.calculators import MACECalculator

            self._property_calculator = MACECalculator(
                models=response_calculator.models,
                model_type="MACEField",
                head=self.head,
                device=self.device,
                default_dtype=self.dtype,
                electric_field=None,
                compute_polarization=True,
                compute_becs=True,
                compute_polarizability=True,
            )
        except (ImportError, TypeError, ValueError, AttributeError) as exc:
            raise RuntimeError(
                "The selected MACE-Field build cannot provide Born charges and "
                "polarizability together"
            ) from exc
        return self._property_calculator

    @staticmethod
    def _calculator_structure(atoms, electric_field=(0.0, 0.0, 0.0)):
        """Copy a structure and attach the explicit field expected by MACE."""

        work = clean_copy(atoms)
        field = np.asarray(electric_field, dtype=float).reshape(-1)
        if field.size != 3 or not np.all(np.isfinite(field)):
            raise ValueError("electric_field must be a finite three-vector")
        work.info = {"electric_field": field.tolist()}
        return work

    def evaluate(
        self,
        structure,
        *,
        electric_field=(0.0, 0.0, 0.0),
        include_response: bool = True,
        include_polarizability: bool = False,
    ) -> ModelEvaluation:
        """Evaluate energy and polarization, with response tensors as requested."""

        require_periodic(structure)
        if include_polarizability and not include_response:
            raise ValueError("polarizability evaluation also requires response tensors")
        work = self._calculator_structure(structure, electric_field)
        calculator = (
            self.make_property_calculator()
            if include_polarizability
            else self.make_calculator()
            if include_response
            else self.make_polarization_calculator()
        )
        properties = ["energy", "polarization"]
        if include_response:
            properties.append("becs")
        if include_polarizability:
            properties.append("polarizability")
        calculator.calculate(work, properties=properties)
        self.evaluation_count += 1
        if include_response:
            self.full_response_evaluations += 1
            becs = _array(calculator.results, "becs", (len(work), 3, 3))
        else:
            self.energy_polarization_evaluations += 1
            becs = np.zeros((len(work), 3, 3), dtype=float)
        polarizability = (
            _array(calculator.results, "polarizability", (3, 3))
            if include_polarizability
            else None
        )
        polarization = _array(calculator.results, "polarization", (3,))
        return ModelEvaluation(
            energy=float(calculator.results["energy"]),
            polarization=polarization,
            reduced_polarization=cartesian_to_reduced(
                polarization, work.get_cell()
            ),
            becs=becs,
            polarizability=polarizability,
        )

    def evaluation_statistics(self) -> dict[str, int]:
        """Return model-call totals for reproducible screening reports."""

        return {
            "total": int(self.evaluation_count),
            "full_response": int(self.full_response_evaluations),
            "energy_polarization": int(self.energy_polarization_evaluations),
        }

    def evaluate_energy_polarization(
        self, structure, *, electric_field=(0.0, 0.0, 0.0)
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate a path image without the Born-charge response."""

        result = self.evaluate(
            structure,
            electric_field=electric_field,
            include_response=False,
        )
        return result.energy, result.polarization, result.reduced_polarization
