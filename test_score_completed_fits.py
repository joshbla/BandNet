import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_triatomic_pipeline import fixture_artifact

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from score_completed_fits import PrefixTrainingArtifact


@unittest.skipUnless(HAS_TORCH, "requires the training extra")
class PrefixTrainingArtifactTests(unittest.TestCase):
    def prefix(self, root, rows):
        """Reduce training bands to the npy header plus `rows` rows, as the restore does."""
        full = root / "train.bands.npy"
        with full.open("rb") as source:
            version = np.lib.format.read_magic(source)
            np.lib.format.read_array_header_1_0(source) if version == (1, 0) else np.lib.format.read_array_header_2_0(source)
            offset = source.tell()
        data = full.read_bytes()
        (root / "train.bands.npy.prefix").write_bytes(data[:offset + rows * 500 * 3 * 8])
        full.unlink()

    def test_verifies_first_training_chunk_and_every_other_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            fixture_artifact(root, checkpoint_rows=4)
            full = np.load(root / "train.bands.npy")
            stop = json.loads((root / "manifest.json").read_text())["splits"]["train"]["chunks"][0]["stop"]
            self.prefix(root, stop + 1)
            artifact = PrefixTrainingArtifact(root)
            labels, bands = artifact.arrays["train"]
            self.assertEqual(len(labels), 8)
            np.testing.assert_array_equal(bands, full[:stop])
            with self.assertRaisesRegex(ValueError, "training prefix"):
                next(artifact.batches("train", 2))
            self.assertEqual(sum(len(rows) for rows, _, _ in artifact.batches("test_dense", 2)), 3)

    def test_rejects_short_or_corrupted_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            fixture_artifact(root, checkpoint_rows=4)
            stop = json.loads((root / "manifest.json").read_text())["splits"]["train"]["chunks"][0]["stop"]
            data = (root / "train.bands.npy").read_bytes()
            self.prefix(root, stop - 1)
            with self.assertRaisesRegex(ValueError, "shorter than its first dataset chunk"):
                PrefixTrainingArtifact(root)
            corrupted = bytearray(data)
            corrupted[-(8 - stop) * 500 * 3 * 8 - 1] ^= 1
            (root / "train.bands.npy").write_bytes(bytes(corrupted))
            self.prefix(root, stop)
            with self.assertRaisesRegex(ValueError, "band checksum"):
                PrefixTrainingArtifact(root)


@unittest.skipUnless(HAS_TORCH, "requires the training extra")
class FitControlTests(unittest.TestCase):
    def test_parallel_workers_may_change_only_thread_count(self):
        from verify_corrected_pilot import fit_controls
        controls = {"device": "cuda:0", "batch_size": 1024, "torch_threads": 8, "checkpoint_steps": 5000, "max_seconds": 60}
        self.assertEqual(fit_controls({"controls": controls}), [controls])
        self.assertEqual(fit_controls({"controls": controls, "fit_scheduling": {"workers": 1}}), [controls])
        worker = dict(controls, torch_threads=4)
        parallel = {"controls": controls, "fit_scheduling": {"workers": 3, "worker_controls": worker}}
        self.assertEqual(fit_controls(parallel), [controls, worker])
        parallel["fit_scheduling"]["worker_controls"] = dict(worker, batch_size=512)
        with self.assertRaisesRegex(ValueError, "beyond thread count"):
            fit_controls(parallel)


if __name__ == "__main__":
    unittest.main()
