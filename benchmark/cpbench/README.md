# CPBench: one pretrained-model experiment

CPBench evaluates a pretrained model and supplied CP method under a required
`Protocol`. The protocol owns partitioning, alpha, split/method seeds, and
assumptions. The result includes metrics, configuration, exact partitions and
provenance. Phase 1 supports multiclass prediction sets.

## Raw TorchCP, shared bindings

From the `pyhealth-fbonc` repository root:

```python
from benchmark.cpbench import PatientSplit, PredictionMethod, Protocol, evaluate
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS

# Supply an already-trained model and held-out samples using its processors.
score = APS(score_type="identity", randomized=True)
predictor = SplitPredictor(score_function=score, device=model.device)
protocol = Protocol(
    split=PatientSplit(calibration_fraction=0.5),  # An explicit experimental choice.
    alpha=0.1,
    split_seed=7,
    method_seed=7,
)
result = evaluate(
    dataset=held_out_samples,
    model=model,
    protocol=protocol,
    method=PredictionMethod(
        fit=predictor.calculate_threshold,
        predict=predictor.predict_with_logits,
        output_kind="multiclass_prediction_set",
        name="torchcp.SplitPredictor/APS",
        config={"score": "APS", "score_type": "identity", "randomized": True},
    ),
)
print(result["metrics"])
```

Change the score to LAC, RAPS, SAPS, TOPK or Margin without adding a wrapper
class. Update the recorded name/configuration to match. The runner imports no
concrete method or TorchCP module.

`PredictionMethod` binds `fit(outputs, labels, *, alpha)` and `predict(outputs)`.
The runner generates model outputs once per sample and passes detached tensors
on the model's device. The callables run with gradients disabled and seeded RNGs.
`fit` calibrates the supplied object in place; use a fresh backend object for
independent concurrent runs. The binding's `config` records already-configured
settings; it does not forward settings to the callables.

The default `input_key="y_prob"` selects probabilities; pair this with an
identity score transformation. To use logits, set `input_key="logit"` and
configure the score to transform logits, for example `score_type="softmax"`.
The model must provide the selected key. Output is a binary `(N, K)` prediction
set with the same class ordering as integer `(N,)` labels.

## Existing native methods

Native PyHealth factories keep their original API:

```python
from pyhealth.calib.predictionset import LABEL

result = evaluate(
    dataset=held_out_samples,
    model=model,
    protocol=protocol,
    method=LABEL,
    method_kwargs={"score_type": "aps"},
)
```

Factories receive `model`, `alpha`, `random_state` and `method_kwargs`; returned
predictors implement `calibrate(cal_dataset=...)` and PyHealth forward outputs
including `loss`, `y_true`, `y_prob`, and `y_predset`. `method_kwargs` is only
for this factory path.

## Protocols

`protocol` is required. There are no top-level `alpha`, `seed`, or
`calibration_fraction` arguments to `evaluate`. `batch_size` remains an execution
setting. Split and method seeds are independent: change `method_seed` to repeat
randomized calibration without changing the partitions.

Choose `SampleSplit()` for sample holdout, `PatientSplit()` for patient holdout,
`RandomSplit(group_by="site_id")` for site/group holdout, or
`TemporalSplit(time_key="timestamp", cutoff=...)` for an earlier/later holdout.
`PredefinedSplit(calibration=[...], test=[...])` reuses exact ordered partitions.
`CustomSplit` or an external policy object handles other selection rules.

`Protocol(disjoint_by=("patient_id", ...))` adds explicit grouping checks.
Patient/group random splits declare their grouping constraint automatically;
sample and temporal splitting do not silently enforce patient separation.
All policies receive basic checks for nonempty, valid, unique, nonoverlapping
sample indices. Unused samples are allowed and recorded.

`result["protocol"]` contains the full protocol specification.
`result["split"]` contains the exact ordered indices, unused indices, declared
groups and a split identifier. This record is returned in memory, not written
to disk. Assumptions are recorded statements, not verified validity guarantees.
See the [protocol guide](../../../cpbench/protocols.md) for the full contract,
custom policies, and replay examples.

## Run the examples

Both examples prepare the same synthetic MLP with PyHealth's `Trainer.train`
before entering CPBench. Processors fit on 12 training patients; 6 separate
validation patients select the best epoch by loss. A further 30 patients form
the evaluation cohort, which the explicit protocol splits into calibration/test.
The helper is `example_utils.prepare_pretrained_example`; no manual optimizer
loop or training inside `evaluate` is used.

Native PyHealth APS (no TorchCP installation required):

```sh
cd /Users/felipe/Desktop/Dev/pyhealth/pyhealth-fbonc
.venv/bin/python -m benchmark.cpbench.pyhealth_example
```

With TorchCP installed, run `python -m benchmark.cpbench.torchcp_example`.
For the isolated trial copy already downloaded on this machine:

```sh
cd /Users/felipe/Desktop/Dev/pyhealth/pyhealth-fbonc
PYTHONPATH="$PWD/benchmark/cpbench/.test-cache/torchcp-deps" \
  .venv/bin/python -m benchmark.cpbench.torchcp_example
```

The complete result prints as JSON; `evaluate` also returns it as a dictionary.
Supply checkpoint identity, backend versions and cohort versions with
`provenance={...}`. This example is an integration check on synthetic data.

The evaluation cohort must exclude the samples/groups used for training and
model selection, as required by the experiment, and reuse the model's fitted
processors/label mapping. CPBench enforces the protocol's declared separation
but cannot infer a pretrained model's history. The dataset exposes `len`,
indexed samples and `subset(indices)`; selected policies require their named
metadata keys. Patient metadata is not required by sample or predefined policies.
The model follows PyHealth's multiclass output conventions.

Both execution paths restore caller RNG state, model training flags and
parameter gradient flags. Reproducibility also requires deterministic model and
backend behavior. Weighted, regression, online, and training-integrated methods
may need additional inputs or lifecycles; they are not automatically supported
by this split-prediction interface. Training orchestration, cached-prediction
inputs, persistence and comparison grids remain outside Phase 1.

Run the focused native/custom/TorchCP tests (TorchCP cases skip if unavailable):

```sh
python -m pytest benchmark/cpbench/test_methods.py benchmark/cpbench/test_protocols.py
```
