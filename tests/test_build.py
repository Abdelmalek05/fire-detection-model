import os
import subprocess
import sys


def _run(args):
    env = dict(os.environ, PYTHONPATH="src")
    return subprocess.run(
        [sys.executable, "-m", "firerisk.build", *args],
        capture_output=True, text=True, env=env,
    )


def test_cli_help_lists_stages():
    r = _run(["--help"])
    assert r.returncode == 0
    for stage in ("firms", "universe", "samples", "weather", "features", "all"):
        assert stage in r.stdout


def test_cli_rejects_unknown_stage():
    r = _run(["nonsense"])
    assert r.returncode != 0
    assert "invalid choice" in (r.stderr + r.stdout).lower()
