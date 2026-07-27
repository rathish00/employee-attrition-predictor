import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition_predictor.bootstrap_pipeline import ensure_model_ready
from attrition_predictor.config import Config
from attrition_predictor.model import AttritionModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestEnsureModelReady(unittest.TestCase):
    """Simulates exactly what a fresh Streamlit Cloud clone sees: source
    code only, no committed data or model artifacts."""

    def setUp(self):
        self.config = Config.load()
        # Save any existing artifacts so this test doesn't destroy local state
        self._backup_dir = PROJECT_ROOT / "tests" / "_bootstrap_backup"
        self._backup_dir.mkdir(exist_ok=True)
        for p in [self.config.paths.raw_data, self.config.paths.model_file, self.config.paths.schema_file]:
            if p.exists():
                shutil.move(str(p), str(self._backup_dir / p.name))

    def tearDown(self):
        for p in [self.config.paths.raw_data, self.config.paths.model_file, self.config.paths.schema_file]:
            p.unlink(missing_ok=True)
        for backed_up in self._backup_dir.iterdir():
            shutil.move(str(backed_up), str(self.config.paths.raw_data.parent / backed_up.name)
                        if backed_up.suffix == ".csv"
                        else str(self.config.paths.model_file.parent / backed_up.name))
        self._backup_dir.rmdir()

    def test_builds_everything_from_nothing(self):
        self.assertFalse(self.config.paths.raw_data.exists())
        self.assertFalse(self.config.paths.model_file.exists())

        messages = []
        model = ensure_model_ready(self.config, on_progress=messages.append)

        self.assertTrue(self.config.paths.raw_data.exists())
        self.assertTrue(self.config.paths.model_file.exists())
        self.assertTrue(self.config.paths.schema_file.exists())
        self.assertIsInstance(model, AttritionModel)
        self.assertGreater(len(messages), 0)

    def test_is_idempotent_and_skips_work_when_artifacts_exist(self):
        ensure_model_ready(self.config)  # first call builds everything
        mtime_before = self.config.paths.model_file.stat().st_mtime

        messages = []
        ensure_model_ready(self.config, on_progress=messages.append)  # second call
        mtime_after = self.config.paths.model_file.stat().st_mtime

        self.assertEqual(mtime_before, mtime_after)  # didn't retrain
        self.assertEqual(messages, ["Loading model..."])  # only the final step ran


if __name__ == "__main__":
    unittest.main()
