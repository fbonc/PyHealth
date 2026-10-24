"""Shared pretrained-model preparation for the runnable CPBench examples."""


def _samples(prefix, patients):
    return [
        {
            "patient_id": f"{prefix}-{patient:03}",
            "visit_id": f"{prefix}-{patient:03}-{visit}",
            "features": [float(patient % 3 == label) for label in range(3)],
            "label": patient % 3,
        }
        for patient in range(patients)
        for visit in range(2)
    ]


def prepare_pretrained_example():
    """Train an MLP with PyHealth Trainer before handing it to CPBench.

    Processors fit only on 12 synthetic training patients. Six other patients
    select the best epoch by validation loss. The returned evaluation cohort
    contains 30 further patients and reuses those fitted processors. Training
    and model selection are entirely outside evaluate(). No files are written.

    Examples:
        >>> model, held_out_samples = prepare_pretrained_example()  # doctest: +SKIP
    """
    import torch

    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.models import MLP
    from pyhealth.trainer import Trainer

    with torch.random.fork_rng():
        torch.manual_seed(3)
        training = create_sample_dataset(
            samples=_samples("train", 12),
            input_schema={"features": "tensor"},
            output_schema={"label": "multiclass"},
        )

        def processed_cohort(prefix, patients):
            return create_sample_dataset(
                samples=_samples(prefix, patients),
                input_schema=training.input_schema,
                output_schema=training.output_schema,
                input_processors=training.input_processors,
                output_processors=training.output_processors,
                dataset_name=f"synthetic-{prefix}",
                task_name="three-class-classification",
            )

        validation = processed_cohort("validation", 6)
        held_out = processed_cohort("held-out", 30)
        model = MLP(training, embedding_dim=8, hidden_dim=8, n_layers=1)
        trainer = Trainer(
            model=model, metrics=["accuracy"], device="cpu", enable_logging=False
        )
        trainer.train(
            train_dataloader=get_dataloader(training, 24, shuffle=True, seed=3),
            val_dataloader=get_dataloader(validation, 12, shuffle=False),
            epochs=5,
            optimizer_params={"lr": 0.03},
            monitor="loss",
            monitor_criterion="min",
            load_best_model_at_last=True,
        )
    model.eval()
    return model, held_out
