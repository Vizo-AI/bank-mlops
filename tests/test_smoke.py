# tests/test_smoke.py
import subprocess, pathlib, sys

def test_script_exists_and_executable():
    script = pathlib.Path("start_mlflow.sh")
    assert script.exists()
    assert script.stat().st_mode & 0o111  # any executable bit

def test_cli_help_runs_quickly():
    out = subprocess.check_output(["bash", "start_mlflow.sh", "--help"],
                                  stderr=subprocess.STDOUT,
                                  timeout=2)
    assert b"mlflow" in out.lower()
