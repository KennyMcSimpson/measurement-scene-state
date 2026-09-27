# Reproduce the three-scene 3D training and mechanism diagnostic

This entry point rebuilds the small `mcss.dynamic.v1` carrier from scratch. Trained weights, downloaded media, environments, and machine-specific output trees are **not included in Git**. The pipeline requires network access for dependencies, pinned calibration metadata, and selected Hypersim archive members. It does not require V5, DINOv2, or other pretrained weights.

## Fresh clone

From a checkout of this repository, install `uv`, use Python 3.11 or 3.12, and initialize the repository's declared submodules. The optional local `map-anything` package is referenced by the dependency configuration; initializing submodules avoids a missing local package path during environment setup.

```bash
git submodule update --init --recursive
uv sync --frozen --extra dev
bash scripts/reproduce_small_3d_pipeline.sh outputs/my-3d-attempt-01 cuda
```

The script locates the repository from its own location, creates a **new** output directory, repeats the frozen environment sync, and logs every stage. Supply `cpu` instead of `cuda` for CPU execution. Relative output paths are resolved from the repository root. A CUDA-capable PyTorch installation and compatible NVIDIA driver are needed for `cuda`; this tiny configuration was exercised on an RTX 5090, but the script does not promise identical timing or bitwise results on different hardware.

The script itself does not update Git or submodules. For its argument summary:

```bash
bash scripts/reproduce_small_3d_pipeline.sh --help
```

## What gets downloaded

Calibration and the upstream license are fetched from Hypersim commit `3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254`, with expected SHA256 checks embedded in the script. Metadata provenance is saved under the new run's `metadata/` directory. See the [upstream Hypersim repository](https://github.com/apple-aiml-research/ml-hypersim) for dataset terms and provenance.

The acquisition command is:

```bash
.venv/bin/python scripts/prepare_small_3d_pilot.py \
  --output outputs/my-3d-attempt-01/data \
  --calibration outputs/my-3d-attempt-01/metadata/metadata_camera_parameters.csv \
  --partitions configs/hypersim_er_partitions.csv
```

This exact preparer permits only `ai_001_001`, `ai_002_001`, and `ai_003_001`, checked against the existing **train** partition. It selects camera `cam_00`, the first 16 jointly available RGB/depth frames, camera poses and scale metadata, and first-frame world positions for calibration verification. It downloads historical preview JPEG RGB and metric ray-distance depth, then prepares 128×160 samples with calibrated intrinsics and a native geometry check. In the recorded run, the selected frame IDs were 0–15. Frame selection is based on archive availability, not image quality or scores; changed or insufficient upstream contents are not silently replaced.

Selected members are fetched through exact HTTP Range requests; full-archive fallback is forbidden. Limits include 64 MiB per range request, a shared 2 GiB response budget per attempt, and a 2 GiB bound on the planned sum of compressed and uncompressed members. The original successful acquisition transferred about 20 MB; this is historical evidence, not a future bandwidth guarantee. Dependency/submodule downloads are separate and can be substantially larger. No dev, diagnostic-test, reserve, or final-holdout scenes are opened.

Prepared manifests contain absolute paths **generated on the current machine**. They work in a fresh checkout without historical home-directory paths, but moving an already prepared run requires regenerating or explicitly rebasing and re-auditing its manifest; copying a historical manifest is not sufficient.

## Training and evaluation commands

The shell entry point writes the exact training configuration: seed `20260927`, 1,000 static steps at `1e-3`, 600 write steps at `3e-4`, gradient clip 1, renderer 64 samples and chunk size 2048. Geometry is explicitly `anchor-centered`, bounds `[-6,-4,-6]` to `[6,4,6]`; omitting that flag selects the old forward-only geometry.

After successful preparation, the equivalent training and audit commands are:

```bash
.venv/bin/python scripts/train_small_3d_pilot.py \
  --manifest outputs/my-3d-attempt-01/data/manifest.json \
  --config outputs/my-3d-attempt-01/training_config.json \
  --spatial-mode anchor-centered --device cuda \
  --output-dir outputs/my-3d-attempt-01/training
.venv/bin/python scripts/audit_small_3d_training.py \
  --training outputs/my-3d-attempt-01/training \
  --expected-a 1000 --expected-b 600 \
  --output outputs/my-3d-attempt-01/training_audit.json
.venv/bin/python scripts/run_trained_3d_mechanism.py \
  --manifest outputs/my-3d-attempt-01/data/manifest.json \
  --checkpoint outputs/my-3d-attempt-01/training/phase_b_final.pt \
  --output outputs/my-3d-attempt-01/mechanism --device cuda
.venv/bin/python -m pytest -q
```

Training starts from the fixed seed; it does not resume a historical checkpoint. `training/` contains `initial.pt`, `phase_a_final.pt`, `phase_b_final.pt`, optimizer/RNG snapshots, locks, access logs, and training/evaluation records. Optimizer snapshots are audit artifacts: this training CLI has no exact-resume flag. The strict checkpoint loader checks both architecture and geometry/config consistency.

Context A is `[0,1,2]`; context B is `[0,3,4]`, sharing only anchor 0. Training streams are `[5,6]`, with supervised training queries `[12,13,14,15]`. The mechanism run uses continuation 7, primary frame-held-out queries `[8,9]`, and a separately reported secondary cohort `[12,13,14,15]`. All 33 candidate states are sealed before evaluator query decoding. Query file hashes and metadata are available for provenance before sealing; this is a numerical information-flow boundary, not an operating-system access sandbox.

## Interpretation and retained failures

All three scenes were used for optimization. Primary queries were withheld from optimizer supervision, but **there are no independent development or test scenes**. This reproduces a post-hoc training-scene diagnostic, not a generalization confirmation. The documented final model still lacks static qualification and a sufficiently trained R-Residual control. Natural-history evaluation and controller training remain skipped. Positive oracle bounds do not establish deployable write gains; numerical FC/CF differences do not establish useful history-conditioned control.

Every pipeline attempt requires a new output directory. On network, calibration, training, or test failure, the script stops and preserves partial files and the stage log. It never deletes a failed attempt, enlarges the data search, retries with altered metrics, or downloads a complete archive. To retry the whole pipeline, choose a new directory, for example `outputs/my-3d-attempt-02`; network budgets apply separately to each attempt. If only tests fail, investigate and record an explicit second test invocation to `pytest_attempt2.log` without erasing the first. Reusing a successful data preparation with new training/evaluation output directories is possible through the commands above. Do not rerun a failed preparer into its existing directory.

The repository's optional legacy V5 checkpoint test may skip when the old V5 artifact is absent; the newly generated dynamic checkpoint is a different architecture and must not be substituted for it.
