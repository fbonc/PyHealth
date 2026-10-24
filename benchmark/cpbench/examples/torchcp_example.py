"""Run raw TorchCP through shared method bindings on a synthetic-data example.

Run from the repository root with TorchCP installed:
    python -m benchmark.cpbench.torchcp_example

The runner does not import this module or require TorchCP.
"""

import json

from benchmark.cpbench import PatientSplit, PredictionMethod, Protocol, evaluate
from benchmark.cpbench.example_utils import prepare_pretrained_example


def main():
    """Run TorchCP APS on a model trained before entering CPBench.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    from torchcp import __version__
    from torchcp.classification.predictor import SplitPredictor
    from torchcp.classification.score import APS

    model, held_out = prepare_pretrained_example()
    # Swap APS for LAC, RAPS, SAPS, TOPK or Margin without adding a wrapper.
    score = APS(score_type="identity", randomized=True)
    predictor = SplitPredictor(score_function=score, device=model.device)
    method = PredictionMethod(
        fit=predictor.calculate_threshold,
        predict=predictor.predict_with_logits,
        output_kind="multiclass_prediction_set",
        name="torchcp.SplitPredictor/APS",
        config={"score": "APS", "score_type": "identity", "randomized": True},
    )
    result = evaluate(
        dataset=held_out,
        model=model,
        method=method,
        protocol=Protocol(
            split=PatientSplit(calibration_fraction=0.5),
            alpha=0.1,
            split_seed=7,
            method_seed=7,
            assumptions=(
                "Evaluation patients excluded from training and model selection",
            ),
        ),
        batch_size=7,
        provenance={
            "torchcp_version": __version__,
            "model_source": "PyHealth Trainer: 12 training and 6 validation patients",
            "model_selection": "best validation loss over 5 epochs; training seed 3",
            "dataset_version": "synthetic-example-v1",
        },
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
