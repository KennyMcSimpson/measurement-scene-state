"""Infrastructure-failure policy of the frozen runner; no model, data or GPU is used."""

import importlib.util
import json
import signal
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals required")

NO_CORE = "import resource\nresource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
CRASH_ONCE = NO_CORE + (
    "import os, pathlib, signal, sys\n"
    "marker, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])\n"
    "work.mkdir()\n"
    "if not marker.exists():\n"
    "    marker.write_text('crashed')\n"
    "    os.kill(os.getpid(), signal.SIGSEGV)\n"
    "print('ok')\n"
)
ALWAYS_CRASH = NO_CORE + "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n"
CUDA_ONCE = (
    "import pathlib, sys\n"
    "marker = pathlib.Path(sys.argv[1])\n"
    "if not marker.exists():\n"
    "    marker.write_text('failed')\n"
    "    print('RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED')\n"
    "    raise SystemExit(1)\n"
    "print('ok')\n"
)


def runner():
    path = Path(__file__).parents[1] / "scripts" / "run_geometry_carrier_experiment.py"
    spec = importlib.util.spec_from_file_location("geometry_carrier_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def experiment(tmp_path):
    root = tmp_path / "exp"
    (root / "audit").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    return root


def script(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_native_crash_is_archived_and_rerun_from_scratch(tmp_path):
    module, root = runner(), experiment(tmp_path)
    work = root / "checkpoints" / "C1_1"
    times = {}
    args = [script(tmp_path, "crash_once.py", CRASH_ONCE), str(tmp_path / "marker"), str(work)]
    module.execute(root, "train_C1_1", args, env=None, times=times, archive=[work])
    incidents = json.loads((root / "audit/incidents.json").read_text())
    assert len(incidents) == 1 and incidents[0]["returncode"] == -signal.SIGSEGV
    assert incidents[0]["kind"] == "native_crash"
    failed = Path(incidents[0]["archived_to"])
    assert (failed / "train_C1_1.log").is_file() and (failed / "C1_1").is_dir()
    assert work.is_dir() and "train_C1_1" in times
    assert (root / "audit/train_C1_1.log").read_text().strip() == "ok"


def test_cuda_resource_error_is_archived_and_retried(tmp_path):
    module, root = runner(), experiment(tmp_path)
    times = {}
    args = [script(tmp_path, "cuda_once.py", CUDA_ONCE), str(tmp_path / "marker")]
    module.execute(root, "phase", args, env=None, times=times, archive=[])
    incidents = json.loads((root / "audit/incidents.json").read_text())
    assert [(i["kind"], i["returncode"]) for i in incidents] == [("cuda_resource", 1)]
    assert "phase" in times


def test_retries_are_bounded_and_counted_across_invocations(tmp_path):
    module, root = runner(), experiment(tmp_path)
    crash = [script(tmp_path, "always.py", ALWAYS_CRASH)]
    with pytest.raises(module.subprocess.CalledProcessError):
        module.execute(root, "phase", crash, env=None, times={}, archive=[], retries=2)
    incidents = json.loads((root / "audit/incidents.json").read_text())
    assert [i["attempt"] for i in incidents] == [1, 2, 3]
    with pytest.raises(PermissionError, match="retry budget"):
        module.execute(root, "phase", crash, env=None, times={}, archive=[], retries=2)
    assert len(json.loads((root / "audit/incidents.json").read_text())) == 3


def test_one_prior_failure_leaves_the_remaining_budget(tmp_path):
    module, root = runner(), experiment(tmp_path)
    (root / "audit/failed_attempts/phase_attempt1").mkdir(parents=True)
    times, calls = {}, []
    ok = [script(tmp_path, "ok.py", "print('ok')\n")]
    module.execute(
        root,
        "phase",
        ok,
        env=None,
        times=times,
        archive=[],
        before_attempt=lambda label, attempt: calls.append((label, attempt)),
    )
    assert "phase" in times and calls == [("phase", 2)]
    assert not (root / "audit/incidents.json").exists()


def test_python_error_is_never_retried(tmp_path):
    module, root = runner(), experiment(tmp_path)
    args = [script(tmp_path, "fail.py", "raise SystemExit(3)\n")]
    with pytest.raises(module.subprocess.CalledProcessError) as error:
        module.execute(root, "phase", args, env=None, times={}, archive=[])
    assert error.value.returncode == 3
    assert not (root / "audit/incidents.json").exists()


def test_gpu_gate_waits_for_frozen_headroom():
    module = runner()
    values = iter([100, 2000, 5000])
    info = module.wait_for_gpu(3072, poll_seconds=0, probe=lambda: next(values))
    assert info["free_mib"] == 5000
    with pytest.raises(TimeoutError):
        module.wait_for_gpu(3072, poll_seconds=0, timeout_seconds=-1, probe=lambda: 1)


def test_secondary_failure_is_recorded_but_primary_failure_stops(tmp_path, monkeypatch):
    module, root = runner(), experiment(tmp_path)
    config = {
        "seeds": [1, 2],
        "variants": ["C0", "C1", "C2"],
        "primary_pair": ["C0", "C1"],
        "parallel_workers": 3,
        "infrastructure_retries": 2,
    }
    started = []

    def fake(root, label, args, **kwargs):
        started.append(label)
        if label.startswith(tuple(fail)):
            raise module.subprocess.CalledProcessError(3, args)

    monkeypatch.setattr(module, "execute", fake)
    fail = ["train_C2_1"]
    module.train_all(root, config, env=None, times={})
    assert list(json.loads((root / "audit/training_failures.json").read_text())) == ["C2_1"]
    assert sorted(started) == sorted(f"train_{v}_{s}" for v in ("C0", "C1", "C2") for s in (1, 2))
    fail, started[:] = ["train_C1_1"], []
    with pytest.raises(RuntimeError, match="Primary training failed"):
        module.train_all(root, config, env=None, times={})
    assert all(label.endswith("_1") for label in started)
