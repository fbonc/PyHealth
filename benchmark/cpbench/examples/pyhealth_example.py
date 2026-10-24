"""Run native PyHealth APS on a model pretrained using PyHealth Trainer.

From the repository root:
    python -m benchmark.cpbench.pyhealth_example

This example does not import or require TorchCP.
"""

import json

from benchmark.cpbench import PatientSplit, Protocol, evaluate
from benchmark.cpbench.example_utils import prepare_pretrained_example


def main():
    """Prepare a pretrained MLP, then benchmark native LABEL with its APS score.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    from pyhealth.calib.predictionset import LABEL

    model, held_out = prepare_pretrained_example()
    protocol = Protocol(
        split=PatientSplit(calibration_fraction=0.5),
        alpha=0.1,
        split_seed=7,
        method_seed=7,
        assumptions=("Evaluation patients excluded from training and model selection",),
    )
    result = evaluate(
        dataset=held_out,
        model=model,
        method=LABEL,
        method_kwargs={"score_type": "aps"},
        protocol=protocol,
        batch_size=7,
        provenance={
            "model_source": "PyHealth Trainer: 12 training and 6 validation patients",
            "model_selection": "best validation loss over 5 epochs; training seed 3",
            "dataset_version": "synthetic-example-v1",
        },
    )

    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
