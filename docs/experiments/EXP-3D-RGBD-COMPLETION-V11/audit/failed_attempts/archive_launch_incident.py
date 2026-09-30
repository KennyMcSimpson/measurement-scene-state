"""Archive the V11 launch failure (missing checkpoints/ directory) with the runner's own helper."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

root = Path("outputs/EXP-3D-RGBD-COMPLETION-V11").resolve()
running = subprocess.run(["pgrep", "-f", "python scripts/[r]un_rgbd_completion_experiment"], capture_output=True)
if running.returncode == 0:
    raise SystemExit("A V11 runner is still alive; refuse to touch its audit files")
spec = importlib.util.spec_from_file_location("v11_run", "scripts/run_rgbd_completion_experiment.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
for variant in ("C0", "C1"):
    for seed in (20260928, 20260929, 20260930):
        label = f"train_{variant}_{seed}"
        text = (root / "audit" / f"{label}.log").read_text()
        if "FileNotFoundError" not in text or f"checkpoints/{variant}_{seed}" not in text:
            raise SystemExit(f"{label}: not the missing-directory launch failure; stop")
        if (root / "checkpoints" / f"{variant}_{seed}").exists():
            raise SystemExit(f"{label}: a run directory exists; stop")
        run.archive_attempt(
            root,
            label,
            1,
            1,
            [root / "checkpoints" / f"{variant}_{seed}"],
            "launch_missing_checkpoints_directory",
        )
failed = root / "audit" / "failed_attempts"
shutil.move(str(root / "audit" / "training_failures.json"), str(failed / "launch_training_failures.json"))
(failed / "LAUNCH_INCIDENT.md").write_text(
    "# V11 launch incident (2026-09-29 18:18 remote time)\n\n"
    "All six training subprocesses stopped at `destination.mkdir(exist_ok=False)` with "
    "FileNotFoundError: the parent `checkpoints/` directory had not been created at launch "
    "(earlier experiments created it in the launch command). No training step ran, no TRAIN, DEV "
    "or EVAL-V3 media was read and no metric was computed.\n\n"
    "The six attempts are archived here by the runner's own `archive_attempt` (incident kind "
    "`launch_missing_checkpoints_directory`, recorded in `audit/incidents.json`) and count against "
    "the frozen infrastructure retry budget. The identical frozen commands are rerun after creating "
    "the directory. No frozen file, code or configuration changed.\n"
)
(root / "checkpoints").mkdir()
incidents = json.loads((root / "audit" / "incidents.json").read_text())
print("archived", len(incidents), sorted(p.name for p in failed.iterdir()))
