"""Explicit experiment protocols and replaceable calibration/test policies."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np


def _fraction(value):
    if isinstance(value, bool) or not 0 < value < 1:
        raise ValueError("calibration_fraction must be strictly between 0 and 1.")


def _seed(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= value < 2**32
    ):
        raise ValueError(f"{name} must be an integer in [0, 2**32).")


def _values(dataset, key):
    """Read scalar grouping/time metadata without requiring patient mappings."""
    values = []
    for index in range(len(dataset)):
        try:
            value = dataset[index][key]
        except KeyError as error:
            raise ValueError(
                f"The split requires metadata key {key!r} on every sample."
            ) from error
        if hasattr(value, "item"):
            value = value.item()
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            raise ValueError(f"Missing or nonfinite {key!r} at sample {index}.")
        try:
            hash(value)
        except TypeError as error:
            raise ValueError(
                f"{key!r} must contain scalar, hashable values."
            ) from error
        values.append(value)
    return values


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(
        f"Protocol metadata must be JSON serializable: {type(value).__name__}"
    )


def _json_record(value):
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


@dataclass(frozen=True)
class SplitIndices:
    """Ordered indices for calibration and test; other samples may be unused.

    Examples:
        >>> parts = SplitIndices(calibration=[0, 2], test=[1, 3])
        >>> list(parts.calibration)
        [0, 2]
    """

    calibration: Sequence[int]
    test: Sequence[int]

    def __post_init__(self):
        object.__setattr__(self, "calibration", tuple(self.calibration))
        object.__setattr__(self, "test", tuple(self.test))


@dataclass(frozen=True)
class RandomSplit:
    """Random holdout by sample or by a caller-selected metadata key.

    Fractions apply to samples when group_by is None, or to whole groups
    otherwise. Group IDs are sorted before shuffling. Within-partition sample
    indices are sorted to preserve efficient reads. No grouping is implicit.

    Examples:
        >>> policy = RandomSplit(group_by="site_id", calibration_fraction=0.5)
    """

    calibration_fraction: float = 0.5
    group_by: str | None = None

    def __post_init__(self):
        _fraction(self.calibration_fraction)
        if self.group_by is not None and (
            not isinstance(self.group_by, str) or not self.group_by
        ):
            raise ValueError("group_by must be None or a nonempty metadata key.")

    @property
    def disjoint_by(self):
        return () if self.group_by is None else (self.group_by,)

    def split(self, dataset, *, seed):
        if self.group_by is None:
            groups = {index: [index] for index in range(len(dataset))}
        else:
            groups = {}
            for index, group in enumerate(_values(dataset, self.group_by)):
                groups.setdefault(group, []).append(index)
        keys = sorted(groups, key=lambda value: (str(value), type(value).__name__))
        order = np.random.default_rng(seed).permutation(len(keys))
        cut = int(len(keys) * self.calibration_fraction)

        def indices(selection):
            return sorted(
                index for position in selection for index in groups[keys[position]]
            )

        return SplitIndices(indices(order[:cut]), indices(order[cut:]))

    def to_dict(self):
        return {
            "kind": "random",
            "calibration_fraction": self.calibration_fraction,
            "group_by": self.group_by,
        }


def PatientSplit(calibration_fraction=0.5):
    """Select patient grouping explicitly as a random split policy.

    Examples:
        >>> policy = PatientSplit(calibration_fraction=0.6)
        >>> policy.group_by
        'patient_id'
    """
    return RandomSplit(calibration_fraction, group_by="patient_id")


def SampleSplit(calibration_fraction=0.5):
    """Select sample-level random partitioning with no patient restriction.

    Examples:
        >>> policy = SampleSplit(calibration_fraction=0.6)
        >>> policy.group_by is None
        True
    """
    return RandomSplit(calibration_fraction)


@dataclass(frozen=True)
class TemporalSplit:
    """Use earlier samples for calibration and later samples for test.

    Calibration timestamps are strictly less than cutoff; test timestamps
    are greater than or equal to it. Each partition is ordered by timestamp,
    then original index. Use comparable values (e.g. datetimes, numeric times,
    or consistent ISO strings). Patient separation is not assumed.

    Examples:
        >>> policy = TemporalSplit(time_key="timestamp", cutoff="2025-01-01")
    """

    time_key: str
    cutoff: Any

    def __post_init__(self):
        if (
            not isinstance(self.time_key, str)
            or not self.time_key
            or self.cutoff is None
        ):
            raise ValueError("TemporalSplit needs a nonempty time_key and a cutoff.")

    def split(self, dataset, *, seed):
        values = _values(dataset, self.time_key)
        ordered = sorted(range(len(dataset)), key=lambda index: (values[index], index))
        return SplitIndices(
            [index for index in ordered if values[index] < self.cutoff],
            [index for index in ordered if values[index] >= self.cutoff],
        )

    def to_dict(self):
        return {"kind": "temporal", "time_key": self.time_key, "cutoff": self.cutoff}


@dataclass(frozen=True)
class PredefinedSplit(SplitIndices):
    """Reuse exact calibration/test indices, independent of the split seed.

    Examples:
        >>> policy = PredefinedSplit(calibration=[2, 0], test=[3, 1])
    """

    def split(self, dataset, *, seed):
        return SplitIndices(self.calibration, self.test)

    def to_dict(self):
        return {
            "kind": "predefined",
            "calibration": list(self.calibration),
            "test": list(self.test),
        }


@dataclass(frozen=True)
class CustomSplit:
    """Bind a custom splitter returning SplitIndices without a registry.

    The callback accepts ``(dataset, *, seed)``. Use its supplied seed with a
    local RNG for reproducibility. Config records its actual selection rules,
    including any deliberate shift or exclusion; it does not execute them.

    Examples:
        >>> policy = CustomSplit(  # doctest: +SKIP
        ...     function=my_splitter, name="held-out-sites", config={"sites": ["B"]}
        ... )
    """

    function: Callable
    name: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if (
            not callable(self.function)
            or not isinstance(self.name, str)
            or not self.name
        ):
            raise ValueError("CustomSplit needs a callable and a nonempty name.")
        object.__setattr__(self, "config", _json_record(dict(self.config)))

    def split(self, dataset, *, seed):
        return self.function(dataset, seed=seed)

    def to_dict(self):
        return {"kind": "custom", "name": self.name, "config": dict(self.config)}


def _check_indices(parts, length):
    result = {}
    for name in ("calibration", "test"):
        indices = list(getattr(parts, name))
        if not indices:
            raise ValueError(f"The protocol must leave {name} nonempty.")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in indices
        ):
            raise ValueError(f"{name} indices must be integers.")
        indices = [int(index) for index in indices]
        if any(index < 0 or index >= length for index in indices):
            raise ValueError(f"{name} indices must be in [0, {length}).")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} contains duplicate sample indices.")
        result[name] = indices
    if set(result["calibration"]) & set(result["test"]):
        raise ValueError("Calibration and test sample indices overlap.")
    return result


@dataclass(frozen=True)
class Protocol:
    """Own a split policy, alpha, independent seeds, and declared assumptions.

    ``split`` implements ``split(dataset, *, seed) -> SplitIndices`` and
    ``to_dict() -> dict``. External policy objects need no registration.
    ``disjoint_by`` adds grouping constraints to the policy's own declarations;
    checks apply only to those explicitly declared keys. Assumptions are
    recorded statements, not statistical guarantees verified by CPBench.

    Examples:
        >>> protocol = Protocol(
        ...     split=PatientSplit(0.5), alpha=0.1, split_seed=7, method_seed=42,
        ...     assumptions=("Evaluation patients excluded from model selection",),
        ... )
    """

    split: Any
    alpha: float = 0.1
    split_seed: int = 42
    method_seed: int = 42
    disjoint_by: Sequence[str] = ()
    assumptions: Sequence[str] = ()
    name: str = "split_conformal"

    def __post_init__(self):
        if isinstance(self.alpha, bool) or not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Protocol name must be a nonempty string.")
        _seed(self.split_seed, "split_seed")
        _seed(self.method_seed, "method_seed")
        if not callable(getattr(self.split, "split", None)) or not callable(
            getattr(self.split, "to_dict", None)
        ):
            raise TypeError("A split policy must implement split() and to_dict().")
        if isinstance(self.disjoint_by, str) or isinstance(self.assumptions, str):
            raise TypeError("disjoint_by and assumptions must be sequences of strings.")
        if any(not isinstance(key, str) or not key for key in self.disjoint_by):
            raise ValueError("disjoint_by must contain nonempty metadata keys.")
        if any(not isinstance(value, str) for value in self.assumptions):
            raise TypeError("assumptions must contain strings.")
        object.__setattr__(self, "disjoint_by", tuple(self.disjoint_by))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        self.to_dict()

    def to_dict(self):
        return _json_record(
            {
                "name": self.name,
                "alpha": float(self.alpha),
                "split_seed": int(self.split_seed),
                "method_seed": int(self.method_seed),
                "disjoint_by": list(self.disjoint_by),
                "assumptions": list(self.assumptions),
                "split": self.split.to_dict(),
            }
        )

    def partition(self, dataset):
        """Execute the declared split and return subsets plus an in-memory manifest."""
        parts = self.split.split(dataset, seed=int(self.split_seed))
        if not isinstance(parts, SplitIndices):
            raise TypeError("The split policy must return SplitIndices.")
        indices = _check_indices(parts, len(dataset))
        keys = tuple(
            dict.fromkeys((*getattr(self.split, "disjoint_by", ()), *self.disjoint_by))
        )
        groups = {}
        for key in keys:
            values = _values(dataset, key)
            selected = {
                name: {values[index] for index in partition}
                for name, partition in indices.items()
            }
            if selected["calibration"] & selected["test"]:
                raise ValueError(f"The protocol requires disjoint {key!r} groups.")
            groups[key] = {
                name: sorted(
                    values, key=lambda value: (str(value), type(value).__name__)
                )
                for name, values in selected.items()
            }
        used = set(indices["calibration"]) | set(indices["test"])
        manifest = _json_record(
            {
                "policy": self.split.to_dict(),
                "seed": int(self.split_seed),
                "partitions": indices,
                "unused": sorted(set(range(len(dataset))) - used),
                "groups": groups,
            }
        )
        manifest["id"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        return (
            dataset.subset(indices["calibration"]),
            dataset.subset(indices["test"]),
            manifest,
        )
