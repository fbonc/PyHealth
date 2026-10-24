"""Check declared splitting semantics independently of any model or backend."""

import json
import unittest
from datetime import UTC, datetime

import numpy as np

from benchmark.cpbench import (
    CustomSplit,
    PatientSplit,
    PredefinedSplit,
    Protocol,
    RandomSplit,
    SampleSplit,
    SplitIndices,
    TemporalSplit,
)


class _Dataset:
    """Minimal external dataset with no patient_to_index mapping."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def subset(self, indices):
        return _Dataset([self.rows[index] for index in indices])


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.dataset = _Dataset(
            [
                {"patient_id": f"p{i // 2}", "site_id": f"s{i // 4}", "timestamp": i}
                for i in range(16)
            ]
        )

    def test_sample_split_needs_no_patient_metadata(self):
        dataset = _Dataset([{"x": i} for i in range(10)])
        protocol = Protocol(split=SampleSplit(0.6), split_seed=7)
        calibration, test, manifest = protocol.partition(dataset)
        self.assertEqual((len(calibration), len(test)), (6, 4))
        self.assertEqual(manifest["groups"], {})
        self.assertEqual(manifest["unused"], [])
        self.assertEqual(manifest, protocol.partition(dataset)[2])
        json.dumps(manifest, allow_nan=False)

    def test_patient_and_site_splits_keep_groups_whole(self):
        for policy, key in (
            (PatientSplit(), "patient_id"),
            (RandomSplit(group_by="site_id"), "site_id"),
        ):
            with self.subTest(key=key):
                _, _, manifest = Protocol(split=policy, split_seed=7).partition(
                    self.dataset
                )
                groups = manifest["groups"][key]
                self.assertTrue(set(groups["calibration"]).isdisjoint(groups["test"]))
                self.assertEqual(
                    sorted(
                        manifest["partitions"]["calibration"]
                        + manifest["partitions"]["test"]
                    ),
                    list(range(len(self.dataset))),
                )

    def test_sample_policy_does_not_impose_patient_disjointness(self):
        dataset = _Dataset([{"patient_id": "shared"} for _ in range(4)])
        _, _, manifest = Protocol(split=SampleSplit()).partition(dataset)
        self.assertEqual(manifest["groups"], {})
        with self.assertRaisesRegex(ValueError, "disjoint 'patient_id'"):
            Protocol(split=SampleSplit(), disjoint_by=("patient_id",)).partition(
                dataset
            )

    def test_changing_method_seed_preserves_split(self):
        first = Protocol(split=PatientSplit(), split_seed=5, method_seed=7)
        second = Protocol(split=PatientSplit(), split_seed=5, method_seed=9)
        self.assertEqual(
            first.partition(self.dataset)[2], second.partition(self.dataset)[2]
        )
        third = Protocol(split=PatientSplit(), split_seed=6, method_seed=7)
        self.assertNotEqual(
            first.partition(self.dataset)[2]["partitions"],
            third.partition(self.dataset)[2]["partitions"],
        )

    def test_patient_assignment_independent_of_row_order(self):
        reordered = _Dataset(list(reversed(self.dataset.rows)))
        protocol = Protocol(split=PatientSplit(), split_seed=8)
        self.assertEqual(
            protocol.partition(self.dataset)[2]["groups"],
            protocol.partition(reordered)[2]["groups"],
        )

    def test_temporal_boundary_ties_and_output_order(self):
        dataset = _Dataset([{"timestamp": value} for value in (5, 1, 3, 3, 2, 4)])
        _, _, manifest = Protocol(split=TemporalSplit("timestamp", 3)).partition(
            dataset
        )
        self.assertEqual(
            manifest["partitions"], {"calibration": [1, 4], "test": [2, 3, 5, 0]}
        )
        self.assertEqual(manifest["groups"], {})

    def test_datetime_cutoff_is_serializable(self):
        cutoff = datetime(2025, 1, 2, tzinfo=UTC)
        dataset = _Dataset(
            [{"time": datetime(2025, 1, i, tzinfo=UTC)} for i in (1, 2, 3)]
        )
        protocol = Protocol(split=TemporalSplit("time", cutoff))
        _, _, manifest = protocol.partition(dataset)
        self.assertEqual(manifest["policy"]["cutoff"], cutoff.isoformat())
        json.dumps(protocol.to_dict(), allow_nan=False)

    def test_predefined_order_unused_samples_and_replay(self):
        protocol = Protocol(split=PredefinedSplit(calibration=[6, 2], test=[9, 7, 0]))
        calibration, test, manifest = protocol.partition(self.dataset)
        self.assertEqual([row["timestamp"] for row in calibration.rows], [6, 2])
        self.assertEqual([row["timestamp"] for row in test.rows], [9, 7, 0])
        self.assertEqual(
            manifest["unused"], [i for i in range(16) if i not in (6, 2, 9, 7, 0)]
        )
        replay = Protocol(
            split=PredefinedSplit(**manifest["partitions"]), split_seed=99
        )
        self.assertEqual(
            manifest["partitions"], replay.partition(self.dataset)[2]["partitions"]
        )

    def test_custom_policy_receives_split_seed_and_records_configuration(self):
        received = []

        def choose(dataset, *, seed):
            received.append(seed)
            order = np.random.default_rng(seed).permutation(len(dataset))
            return SplitIndices(order[:3], order[3:6])

        policy = CustomSplit(choose, name="subsample", config={"size": 6})
        protocol = Protocol(
            split=policy,
            split_seed=11,
            method_seed=22,
            assumptions=("declared assumption",),
        )
        _, _, manifest = protocol.partition(self.dataset)
        self.assertEqual(received, [11])
        self.assertEqual(manifest["policy"]["config"], {"size": 6})
        self.assertEqual(protocol.to_dict()["assumptions"], ["declared assumption"])

    def test_external_policy_needs_no_inheritance_or_registration(self):
        class ExternalPolicy:
            def split(self, dataset, *, seed):
                return SplitIndices([0, 1], [2, 3])

            def to_dict(self):
                return {"kind": "external"}

        self.assertEqual(
            Protocol(split=ExternalPolicy()).partition(self.dataset)[2]["policy"],
            {"kind": "external"},
        )

    def test_bad_partitions_fail_before_subsetting(self):
        cases = [
            ([0], [0], "overlap"),
            ([0, 0], [1], "duplicate"),
            ([], [1], "nonempty"),
            ([-1], [1], "must be in"),
            ([0], [100], "must be in"),
            ([True], [1], "integers"),
            ([0.0], [1], "integers"),
        ]
        for calibration, test, error in cases:
            with self.subTest(calibration=calibration, test=test):
                protocol = Protocol(split=PredefinedSplit(calibration, test))
                with self.assertRaisesRegex(ValueError, error):
                    protocol.partition(self.dataset)
        with self.assertRaisesRegex(TypeError, "SplitIndices"):
            Protocol(
                split=CustomSplit(lambda dataset, *, seed: ([0], [1]), "bad")
            ).partition(self.dataset)

    def test_invalid_protocol_configuration(self):
        for alpha in (0, 1, float("nan"), True):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(ValueError, "alpha"):
                Protocol(split=SampleSplit(), alpha=alpha)
        for kwargs in (
            {"split_seed": -1},
            {"method_seed": 2**32},
            {"split_seed": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Protocol(split=SampleSplit(), **kwargs)
        with self.assertRaisesRegex(TypeError, "split policy"):
            Protocol(split=lambda dataset: None)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            Protocol(split=PatientSplit()).partition(_Dataset([{"patient_id": "only"}]))


if __name__ == "__main__":
    unittest.main()
