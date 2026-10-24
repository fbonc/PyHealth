"""Check shared prediction bindings and compatibility with native methods."""

import importlib.util
import json
import random
import unittest
from functools import partial
from unittest.mock import patch

import numpy as np
import torch

from benchmark.cpbench import (
    PatientSplit,
    PredictionMethod,
    Protocol,
    SampleSplit,
    evaluate,
)
from benchmark.cpbench.example_utils import prepare_pretrained_example


class TestPredictionMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.dataset = prepare_pretrained_example()
        cls.model.eval()

    def run_method(self, method, **kwargs):
        settings = {
            "dataset": self.dataset,
            "model": self.model,
            "method": method,
            "protocol": Protocol(
                split=PatientSplit(), alpha=0.2, split_seed=7, method_seed=7
            ),
            "batch_size": 7,
        }
        settings.update(kwargs)
        return evaluate(**settings)

    def test_callbacks_receive_predictions_alpha_and_ordered_labels(self):
        captured = {}
        seen = []

        def fit(outputs, labels, *, alpha):
            captured.update(
                calibration=outputs.clone(), labels=labels.clone(), alpha=alpha
            )

        def predict(outputs):
            captured["test"] = outputs.clone()
            return torch.ones_like(outputs, dtype=torch.bool)

        method = PredictionMethod(
            fit=fit,
            predict=predict,
            output_kind="multiclass_prediction_set",
            name="external-method",
            config={"setting": "all-classes"},
        )
        handle = self.model.register_forward_hook(
            lambda module, args, output: seen.append(len(output["y_true"]))
        )
        try:
            result = self.run_method(method)
        finally:
            handle.remove()
        self.assertEqual(sum(seen), len(self.dataset))
        self.assertEqual(captured["alpha"], 0.2)
        self.assertFalse(captured["calibration"].requires_grad)
        self.assertEqual(captured["calibration"].device, self.model.device)
        self.assertEqual(captured["labels"].dtype, torch.int64)
        self.assertEqual(result["metrics"], {"coverage": 1.0, "avg_set_size": 3.0})
        self.assertEqual(result["cp_method"], "external-method")
        self.assertEqual(
            result["config"]["prediction_method"]["parameters"],
            {"setting": "all-classes"},
        )
        patients = result["split"]["groups"]["patient_id"]
        self.assertTrue(set(patients["calibration"]).isdisjoint(patients["test"]))
        self.assertEqual(
            len(captured["test"]), len(result["split"]["partitions"]["test"])
        )
        json.dumps(result, allow_nan=False)

    def test_selecting_logits_and_numpy_prediction_sets(self):
        from pyhealth.datasets import get_dataloader

        captured = {}

        def fit(outputs, labels, *, alpha):
            captured["calibration"] = outputs.clone()

        method = PredictionMethod(
            fit=partial(fit),
            predict=lambda outputs: np.ones(tuple(outputs.shape), dtype=bool),
            output_kind="multiclass_prediction_set",
            input_key="logit",
        )
        result = self.run_method(method)
        subset = self.dataset.subset(result["split"]["partitions"]["calibration"])
        with torch.no_grad():
            expected = torch.cat(
                [self.model(**batch)["logit"] for batch in get_dataloader(subset, 7)]
            )
        torch.testing.assert_close(captured["calibration"], expected)
        self.assertEqual(result["config"]["prediction_method"]["input_key"], "logit")
        self.assertEqual(result["metrics"]["coverage"], 1.0)

    def test_protocol_is_required_and_controls_evaluation(self):
        method = PredictionMethod(
            lambda outputs, labels, *, alpha: None,
            lambda outputs: torch.ones_like(outputs, dtype=torch.bool),
            "multiclass_prediction_set",
        )
        with self.assertRaisesRegex(TypeError, "protocol"):
            evaluate(dataset=self.dataset, model=self.model, method=method)
        with self.assertRaisesRegex(TypeError, "alpha"):
            self.run_method(method, alpha=0.1)
        result = self.run_method(
            method,
            protocol=Protocol(
                split=SampleSplit(0.6), alpha=0.3, split_seed=5, method_seed=9
            ),
        )
        self.assertEqual(result["alpha"], 0.3)
        self.assertEqual(result["protocol"]["split_seed"], 5)
        self.assertEqual(result["seed"], 9)
        self.assertEqual(result["split"]["groups"], {})
        self.assertEqual(len(result["split"]["partitions"]["calibration"]), 36)

    def test_failure_restores_model_and_random_states(self):
        def fail(outputs, labels, *, alpha):
            random.random()
            np.random.random()
            torch.rand(3)
            self.model.train()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            raise RuntimeError("fit failed")

        method = PredictionMethod(
            fail, lambda outputs: outputs, "multiclass_prediction_set"
        )
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        flags = [p.requires_grad for p in self.model.parameters()]
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            self.run_method(method)
        self.assertFalse(self.model.training)
        self.assertEqual(flags, [p.requires_grad for p in self.model.parameters()])
        self.assertEqual(python_state, random.getstate())
        current_numpy = np.random.get_state()
        np.testing.assert_array_equal(numpy_state[1], current_numpy[1])
        self.assertEqual(numpy_state[2:], current_numpy[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_native_factory_still_works(self):
        from pyhealth.calib.predictionset import LABEL

        result = self.run_method(LABEL, method_kwargs={"score_type": "aps"})
        self.assertTrue(result["cp_method"].endswith(".LABEL"))
        self.assertEqual(
            result, self.run_method(LABEL, method_kwargs={"score_type": "aps"})
        )

    def test_unsupported_contract_and_malformed_outputs_fail_clearly(self):
        fit = lambda outputs, labels, *, alpha: None
        method = PredictionMethod(
            fit, lambda outputs: outputs, "multiclass_prediction_set"
        )
        with self.assertRaisesRegex(ValueError, "binary y_predset"):
            self.run_method(method)
        with self.assertRaisesRegex(ValueError, "method_kwargs"):
            self.run_method(method, method_kwargs={"randomized": False})
        intervals = PredictionMethod(
            fit, lambda outputs: outputs, "regression_interval"
        )
        with self.assertRaisesRegex(NotImplementedError, "multiclass"):
            self.run_method(intervals)

    @unittest.skipUnless(importlib.util.find_spec("torchcp"), "TorchCP is optional")
    def test_raw_torchcp_scores_share_one_binding(self):
        from torchcp.classification.predictor import SplitPredictor
        from torchcp.classification.score import APS, LAC, RAPS, SAPS, TOPK, Margin

        for score_cls in (LAC, APS, RAPS, SAPS, TOPK, Margin):
            with self.subTest(score=score_cls.__name__):
                score = score_cls(score_type="identity")
                predictor = SplitPredictor(score, alpha=0.4, device=self.model.device)
                method = PredictionMethod(
                    fit=predictor.calculate_threshold,
                    predict=predictor.predict_with_logits,
                    output_kind="multiclass_prediction_set",
                    name=f"torchcp/{score_cls.__name__}",
                )
                with patch.object(
                    self.model,
                    "forward",
                    wraps=self.model.forward,
                ) as forward:
                    first = self.run_method(method)
                self.assertEqual(
                    sum(
                        len(call.kwargs["patient_id"])
                        for call in forward.call_args_list
                    ),
                    len(self.dataset),
                )
                self.assertEqual(first, self.run_method(method))
                self.assertTrue(0 <= first["metrics"]["coverage"] <= 1)
                self.assertTrue(0 <= first["metrics"]["avg_set_size"] <= 3)
                # evaluate's alpha overrides the predictor's constructor default.
                with patch.object(
                    predictor,
                    "calculate_threshold",
                    wraps=predictor.calculate_threshold,
                ) as fit:
                    self.run_method(
                        PredictionMethod(
                            fit,
                            predictor.predict_with_logits,
                            "multiclass_prediction_set",
                            name="alpha-check",
                        )
                    )
                self.assertEqual(fit.call_args.kwargs["alpha"], 0.2)

    @unittest.skipUnless(importlib.util.find_spec("torchcp"), "TorchCP is optional")
    def test_class_conditional_predictor_uses_the_same_binding(self):
        from torchcp.classification.predictor import ClassConditionalPredictor
        from torchcp.classification.score import LAC

        predictor = ClassConditionalPredictor(
            LAC(score_type="identity"), device=self.model.device
        )
        method = PredictionMethod(
            predictor.calculate_threshold,
            predictor.predict_with_logits,
            "multiclass_prediction_set",
        )
        result = self.run_method(method)
        self.assertTrue(result["cp_method"].endswith("ClassConditionalPredictor"))
        self.assertEqual(tuple(predictor.q_hat.shape), (3,))


if __name__ == "__main__":
    unittest.main()
