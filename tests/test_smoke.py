"""End-to-end smoke test: trains a model and verifies outputs."""

import os
import subprocess
import sys
import tempfile
import shutil

import pytest

from tests.conftest import DEFAULT_DATA_PATH


@pytest.fixture
def output_dir():
    """Create a temp directory for smoke test outputs, clean up after."""
    d = tempfile.mkdtemp(prefix="equity_smoke_")
    yield d
    shutil.rmtree(d)


def test_train_pipeline_produces_outputs(output_dir):
    """Smoke test: train.py runs end-to-end and produces model + comparison CSV."""
    data_path = os.environ.get("EQUITY_DATA", DEFAULT_DATA_PATH)

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(__file__))

    result = subprocess.run(
        [
            sys.executable, "scripts/train.py",
            "--data", data_path,
            "--models", "ridge",
            "--features", "v1",
            "--target", "overnight_return",
            "--output-dir", output_dir,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        timeout=120,
    )

    assert result.returncode == 0, f"train.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    # Verify model file was saved in temp output dir
    assert os.path.exists(os.path.join(output_dir, "models", "ridge_v1_overnight_return.pkl"))

    # Verify comparison CSV was saved
    assert os.path.exists(os.path.join(output_dir, "model_comparison_overnight_return.csv"))

    # Verify figures were saved
    assert os.path.exists(os.path.join(output_dir, "figures", "ridge_v1_overnight_return_predictions.png"))

    # Stdout should contain CV metrics and baseline
    assert "Out-of-sample CV Metrics" in result.stdout
    assert "MODEL COMPARISON" in result.stdout
