"""Native-crash policy of the frozen runner; no model, data or GPU is used."""

import importlib.util
import json
import signal
from pathlib import Path

import pytest


def runner():
    path = Path(__file__).parents[1] / "scripts" / "run_geometry_carrier_experiment.py"
    spec = importlib.util.spec_from_file_location("geometry_carrier_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CRASH_ONCE = (
    "import os, pathlib, signal, sys\n"
    "marker, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])\n"
    "work.mkdir()\n"
    "if not marker.exists():\n"
    "    marker.write_text('crashed')\n"
    "    os.kill(os.getpid(), signal.SIGSEGV)\n"
    "print('ok')\n"
)


def experiment(tmp_path):
    root = tmp_path / "exp"
    (root / "audit").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    return root


def test_native_crash_is_archived_and_rerun_once_from_scratch(tmp_path):
    module, root = runner(), experiment(tmp_path)
    script = tmp_path / "crash_once.py"
    script.write_text(CRASH_ONCE)
    work = root / "checkpoints" / "C1_1"
    times = {}
    args = [str(script), str(tmp_path / "marker"), str(work)]
    module.execute(root, "train_C1_1", args, env=None, times=times, archive=[work])
    incidents = json.loads((root / "audit/incidents.json").read_text())
    assert len(incidents) == 1 and incidents[0]["returncode"] == -signal.SIGSEGV
    failed = Path(incidents[0]["archived_to"])
    assert (failed / "train_C1_1.log").is_file() and (failed / "C1_1").is_dir()
    assert work.is_dir() and "train_C1_1" in times
    assert (root / "audit/train_C1_1.log").read_text().strip() == "ok"


def test_second_native_crash_stops_after_archiving_both(tmp_path):
    module, root = runner(), experiment(tmp_path)
    script = tmp_path / "always.py"
    script.write_text("import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n")
    with pytest.raises(module.subprocess.CalledProcessError):
        module.execute(root, "phase", [str(script)], env=None, times={}, archive=[])
    incidents = json.loads((root / "audit/incidents.json").read_text())
    assert [i["attempt"] for i in incidents] == [1, 2]


def test_python_error_is_never_retried(tmp_path):
    module, root = runner(), experiment(tmp_path)
    script = tmp_path / "fail.py"
    script.write_text("raise SystemExit(3)\n")
    with pytest.raises(module.subprocess.CalledProcessError) as error:
        module.execute(root, "phase", [str(script)], env=None, times={}, archive=[])
    assert error.value.returncode == 3
    assert not (root / "audit/incidents.json").exists()
