"""Backend-independent bindings for prediction-based conformal methods."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredictionMethod:
    """Bind existing calibration and prediction functions without subclassing.

    ``fit(outputs, labels, *, alpha)`` calibrates the supplied method in place.
    ``predict(outputs)`` returns prediction sets in the same sample/class order.
    Both receive detached tensors on the pretrained model's device. The runner
    seeds their execution; neither callable needs a model or dataset argument.

    Args:
        fit: Calibration callable accepting the experiment's ``alpha`` keyword.
        predict: Callable producing sets from model outputs.
        output_kind: Explicit result semantics. Phase 1 supports only
            ``"multiclass_prediction_set"``.
        input_key: Model output to collect, e.g. ``"y_prob"`` or ``"logit"``.
            Match the method's input transformation to this choice.
        name: Optional method identity for results. Otherwise inferred from fit.
        config: Serializable algorithm settings recorded with results. These
            describe the already-configured callables; they are not forwarded.

    Examples:
        >>> method = PredictionMethod(  # doctest: +SKIP
        ...     fit=predictor.calculate_threshold,
        ...     predict=predictor.predict_with_logits,
        ...     output_kind="multiclass_prediction_set",
        ...     name="my-split-method",
        ... )
    """

    fit: Callable
    predict: Callable
    output_kind: str
    input_key: str = "y_prob"
    name: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not callable(self.fit) or not callable(self.predict):
            raise TypeError("PredictionMethod fit and predict must be callable.")
        if not isinstance(self.input_key, str) or not self.input_key:
            raise ValueError("PredictionMethod input_key must be a nonempty string.")
        object.__setattr__(self, "config", dict(self.config))
