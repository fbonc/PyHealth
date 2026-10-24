"""One pretrained-model conformal experiment under a supplied protocol."""

import random
from contextlib import contextmanager

import numpy as np

from .methods import PredictionMethod
from .protocols import Protocol


@contextmanager
def _experiment_state(model, seed):
    """Seed the experiment and restore caller RNG/model settings on exit."""
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(param, param.requires_grad) for param in model.parameters()]
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes:
            module.training = training
        for param, requires_grad in gradients:
            param.requires_grad_(requires_grad)


def _validate_inputs(
    dataset,
    model,
    method,
    protocol,
    batch_size,
    method_kwargs,
):
    """Validate experiment inputs and copy the method's extra arguments."""
    if not isinstance(protocol, Protocol):
        raise TypeError("protocol must be an explicit Protocol instance.")
    if not callable(getattr(dataset, "subset", None)):
        raise TypeError("dataset must expose subset(indices).")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")
    if getattr(model, "mode", None) != "multiclass":
        raise NotImplementedError("Phase 1 evaluates multiclass prediction sets.")
    method_kwargs = dict(method_kwargs or {})
    if isinstance(method, PredictionMethod):
        if method.output_kind != "multiclass_prediction_set":
            raise NotImplementedError("Phase 1 evaluates multiclass prediction sets.")
        if method_kwargs:
            raise ValueError(
                "Configure PredictionMethod callables directly; method_kwargs "
                "is only for predictor factories."
            )
    elif not callable(method):
        raise TypeError("method must be a PredictionMethod or a predictor factory.")
    if {"model", "alpha", "random_state"} & method_kwargs.keys():
        raise ValueError(
            "Pass model through evaluate and alpha/seeds through Protocol, not method_kwargs."
        )
    return method_kwargs


def _partition_dataset(dataset, protocol):
    """Execute the protocol's policy without selecting a splitting strategy."""
    return protocol.partition(dataset)


def _calibrate_method(method, model, calibration_data, alpha, seed, method_kwargs):
    """Construct the caller's predictor and calibrate it on held-out patients."""
    predictor = method(
        model=model, alpha=float(alpha), random_state=seed, **method_kwargs
    )
    if not callable(getattr(predictor, "calibrate", None)):
        raise TypeError("The method factory must return a predictor with calibrate().")
    predictor.calibrate(cal_dataset=calibration_data)
    return predictor


def _run_inference(predictor, test_data, device, batch_size):
    """Collect test labels, probabilities and prediction sets through PyHealth."""
    from pyhealth.datasets import get_dataloader
    from pyhealth.trainer import Trainer

    loader = get_dataloader(test_data, batch_size=batch_size, shuffle=False)
    trainer = Trainer(model=predictor, device=device, enable_logging=False)
    labels, probabilities, _, outputs = trainer.inference(
        loader, additional_outputs=["y_predset"]
    )
    return labels, probabilities, outputs["y_predset"]


def _collect_predictions(model, dataset, batch_size, input_key):
    """Collect a chosen output from the supplied pretrained model, in order."""
    from pyhealth.datasets import get_dataloader
    from pyhealth.trainer import Trainer

    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    trainer = Trainer(model=model, device=model.device, enable_logging=False)
    if input_key == "y_prob":
        labels, predictions, _ = trainer.inference(loader)
    else:
        labels, _, _, outputs = trainer.inference(
            loader, additional_outputs=[input_key]
        )
        predictions = outputs[input_key]
    return labels, predictions


def _run_prediction_method(
    method, model, calibration_data, test_data, alpha, batch_size
):
    """Run the shared prediction-based lifecycle with caller-supplied functions."""
    import torch

    labels, predictions = _collect_predictions(
        model, calibration_data, batch_size, method.input_key
    )
    with torch.no_grad():
        method.fit(
            torch.as_tensor(predictions, device=model.device),
            torch.as_tensor(labels, device=model.device),
            alpha=float(alpha),
        )
    test_labels, test_predictions = _collect_predictions(
        model, test_data, batch_size, method.input_key
    )
    with torch.no_grad():
        prediction_sets = method.predict(
            torch.as_tensor(test_predictions, device=model.device)
        )
    if torch.is_tensor(prediction_sets):
        prediction_sets = prediction_sets.detach().cpu().numpy()
    return test_labels, test_predictions, prediction_sets


def _method_name(method):
    """Identify supplied bindings or a native predictor without backend imports."""
    if isinstance(method, PredictionMethod):
        if method.name:
            return method.name
        owner = getattr(method.fit, "__self__", None)
        if owner is None:
            module = getattr(method.fit, "__module__", type(method.fit).__module__)
            name = getattr(method.fit, "__qualname__", type(method.fit).__qualname__)
            return f"{module}.{name}"
        method = owner
    return f"{type(method).__module__}.{type(method).__qualname__}"


def _compute_metrics(labels, probabilities, prediction_sets, n_samples):
    """Validate multiclass outputs and compute marginal coverage and set size."""
    from pyhealth.metrics.prediction_set import miscoverage_overall_ps, size

    prediction_sets = np.asarray(prediction_sets)
    labels = np.asarray(labels)
    if (
        prediction_sets.ndim != 2
        or prediction_sets.shape != probabilities.shape
        or prediction_sets.shape[0] != n_samples
        or labels.shape != (n_samples,)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any(labels < 0)
        or np.any(labels >= prediction_sets.shape[1])
        or not np.isin(prediction_sets, [0, 1]).all()
    ):
        raise ValueError(
            "Expected integer class labels (N,) and binary y_predset (N, K) "
            "matching y_prob for every test sample."
        )
    return {
        "coverage": float(1 - miscoverage_overall_ps(prediction_sets, labels)),
        "avg_set_size": float(size(prediction_sets)),
    }


def _build_result(
    *,
    dataset,
    model,
    predictor,
    protocol,
    metrics,
    split,
    batch_size,
    method_kwargs,
    provenance,
):
    """Combine measured metrics with experiment configuration and provenance."""
    import torch

    from pyhealth import __version__

    config = {"batch_size": batch_size, "method_kwargs": method_kwargs}
    if isinstance(predictor, PredictionMethod):
        config["prediction_method"] = {
            "input_key": predictor.input_key,
            "parameters": dict(predictor.config),
        }

    return {
        "dataset": getattr(dataset, "dataset_name", "") or type(dataset).__name__,
        "task": getattr(dataset, "task_name", ""),
        "model": f"{type(model).__module__}.{type(model).__qualname__}",
        "cp_method": _method_name(predictor),
        "output_kind": "multiclass_prediction_set",
        "alpha": float(protocol.alpha),
        "seed": int(protocol.method_seed),
        "protocol": protocol.to_dict(),
        "metrics": metrics,
        "metric_definitions": {
            "coverage": "Fraction of test samples whose set contains the true class.",
            "avg_set_size": "Mean number of classes in each test prediction set.",
        },
        "split": split,
        "config": config,
        "provenance": dict(provenance or {}),
        "versions": {
            "pyhealth": __version__,
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
    }


def evaluate(
    *,
    dataset,
    model,
    method,
    protocol,
    batch_size=32,
    method_kwargs=None,
    provenance=None,
):
    """Evaluate one supplied pretrained model and CP method under a protocol.

    Args:
        dataset: Task-processed evaluation data exposing len, indexed samples,
            and subset(indices). Group/time metadata is required only by the
            selected policy. Reuse the pretrained model's fitted processors.
        model: Supplied pretrained PyHealth-compatible multiclass model.
        method: A PredictionMethod binding existing fit/predict functions,
            or a native predictor factory. No concrete algorithm is selected
            or imported by CPBench.
        protocol: Required Protocol declaring split policy, alpha, independent
            split/method seeds, grouping constraints, and assumptions. CPBench
            executes this policy and never chooses a split strategy itself.
        batch_size: Inference batch size; recorded as execution configuration.
        method_kwargs: Extra arguments for native predictor factories only.
        provenance: Checkpoint, cohort/version and model-selection history.
            CPBench cannot infer or verify a pretrained model's history.

    Returns:
        Metrics, protocol specification, exact ordered split manifest,
        execution configuration, component identities and supplied provenance.
        The manifest is returned in memory; no files are written automatically.

    Examples:
        >>> from benchmark.cpbench import PatientSplit, Protocol, evaluate
        >>> result = evaluate(  # doctest: +SKIP
        ...     dataset=held_out_samples, model=pretrained_model,
        ...     method=my_method_factory,
        ...     protocol=Protocol(split=PatientSplit(), alpha=0.1),
        ... )
    """
    method_kwargs = _validate_inputs(
        dataset, model, method, protocol, batch_size, method_kwargs
    )
    calibration_data, test_data, split = _partition_dataset(dataset, protocol)
    with _experiment_state(model, int(protocol.method_seed)):
        if isinstance(method, PredictionMethod):
            predictor = method
            labels, probabilities, prediction_sets = _run_prediction_method(
                method, model, calibration_data, test_data, protocol.alpha, batch_size
            )
        else:
            predictor = _calibrate_method(
                method,
                model,
                calibration_data,
                protocol.alpha,
                int(protocol.method_seed),
                method_kwargs,
            )
            labels, probabilities, prediction_sets = _run_inference(
                predictor, test_data, model.device, batch_size
            )
    metrics = _compute_metrics(
        labels, probabilities, prediction_sets, len(split["partitions"]["test"])
    )
    return _build_result(
        dataset=dataset,
        model=model,
        predictor=predictor,
        protocol=protocol,
        metrics=metrics,
        split=split,
        batch_size=batch_size,
        method_kwargs=method_kwargs,
        provenance=provenance,
    )
