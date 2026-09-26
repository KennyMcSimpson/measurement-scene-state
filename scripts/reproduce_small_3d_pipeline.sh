#!/usr/bin/env bash
# Fresh TRAIN-only reproduction. Never overwrite an earlier attempt.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  printf '%s\n' 'Usage: bash scripts/reproduce_small_3d_pipeline.sh [NEW_OUTPUT_DIRECTORY] [cuda|cpu]' \
    'Downloads pinned calibration/license and selected members of three TRAIN scenes,' \
    'trains 1000 A + 600 B steps, evaluates fixed diagnostic queries, and runs tests.'
  exit 0
fi
if (( $# > 2 )); then
  printf '%s\n' 'Too many arguments; use --help.' >&2
  exit 2
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
run_dir=${1:-outputs/reproduction-3d-$(date -u +%Y%m%dT%H%M%SZ)}
device=${2:-cuda}
if [[ "$device" != cuda && "$device" != cpu ]]; then
  printf '%s\n' 'Device must be cuda or cpu.' >&2
  exit 2
fi
if [[ -e "$run_dir" ]]; then
  printf 'Refusing existing output: %s\n' "$run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd)
export PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uv sync --frozen --extra dev 2>&1 | tee "$run_dir/environment.log"
python_bin="$repo_root/.venv/bin/python"
"$python_bin" - "$run_dir" <<'PY' 2>&1 | tee "$run_dir/metadata.log"
import hashlib
import json
import sys
from pathlib import Path
import requests

output = Path(sys.argv[1]) / "metadata"
output.mkdir()
commit = "3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254"
base = f"https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/{commit}/"
files = [
    ("contrib/mikeroberts3000/metadata_camera_parameters.csv", "0b40e0884ba2458e7bdcdbb27de46a4ec6eacbfbbb4ef0d70188d78098cebdc9"),
    ("LICENSE.txt", "8bfda405dc86c28a7b8d1a669d17c07976e4a6813ac643a07999d71bd8ee882b"),
]
sources = []
for relative, expected in files:
    url = base + relative
    with requests.get(url, timeout=60, stream=True) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(65536):
            data.extend(chunk)
            if len(data) > 1024 * 1024:
                raise ValueError("Metadata response exceeds 1 MiB limit")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"Pinned metadata hash mismatch: {relative}")
    path = output / Path(relative).name
    path.write_bytes(data)
    sources.append({"url": url, "file": path.name, "sha256": actual, "bytes": len(data)})
(output / "sources.json").write_text(json.dumps({"commit": commit, "files": sources}, indent=2) + "\n")
print("Pinned calibration and license verified.")
PY
"$python_bin" scripts/prepare_small_3d_pilot.py \
  --output "$run_dir/data" \
  --calibration "$run_dir/metadata/metadata_camera_parameters.csv" \
  --partitions configs/hypersim_er_partitions.csv 2>&1 | tee "$run_dir/prepare.log"
cat > "$run_dir/training_config.json" <<'JSON'
{
  "seed": 20260927,
  "stage_a_steps": 1000,
  "stage_b_steps": 600,
  "stage_a_lr": 0.001,
  "stage_b_lr": 0.0003,
  "renderer_samples": 64,
  "ray_chunk_size": 2048,
  "gradient_clip": 1.0
}
JSON
"$python_bin" scripts/train_small_3d_pilot.py \
  --manifest "$run_dir/data/manifest.json" \
  --config "$run_dir/training_config.json" --spatial-mode anchor-centered \
  --output-dir "$run_dir/training" --device "$device" 2>&1 | tee "$run_dir/training.log"
"$python_bin" scripts/audit_small_3d_training.py \
  --training "$run_dir/training" --expected-a 1000 --expected-b 600 \
  --output "$run_dir/training_audit.json" 2>&1 | tee "$run_dir/training_audit.log"
"$python_bin" scripts/run_trained_3d_mechanism.py \
  --manifest "$run_dir/data/manifest.json" \
  --checkpoint "$run_dir/training/phase_b_final.pt" \
  --output "$run_dir/mechanism" --device "$device" 2>&1 | tee "$run_dir/mechanism.log"
"$python_bin" -m pytest -q 2>&1 | tee "$run_dir/pytest_attempt1.log"
printf 'Reproduction complete: %s\n' "$run_dir"
