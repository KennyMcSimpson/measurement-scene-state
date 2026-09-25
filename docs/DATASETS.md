# Dataset Setup

## Replica

Official repository: <https://github.com/facebookresearch/Replica-Dataset>

The official v1.0 Windows release is 17 concatenated gzip-tar parts (`partaa` through `partaq`).
The current downloader resumes each part and can stream-extract the concatenated archive without
creating a second merged tarball. The extracted package contains meshes, textures, and Habitat
configuration. It does not by itself define the sparse RGB-D camera trajectories used by this
project.

Preview the exact plan without writing:

```powershell
uv run mcss download replica --root data\replica_raw --dry-run
```

Download and extract after D: has enough space:

```powershell
uv run mcss download replica `
  --root data\replica_raw `
  --extract-to data\replica_assets
```

`prepare replica` expects a rendered sequence with matching RGB/depth files, `traj.txt`, and
`cam_params.json`. If the official mesh is the only available input, render a deterministic
trajectory with Habitat-Sim first, then pass that rendered directory to:

```powershell
uv run mcss prepare replica `
  --raw-root data\replica_rendered `
  --output-root data\replica_prepared `
  --image-height 128 --image-width 160
```

Do not treat a mesh-only Replica download as a completed training dataset. The adapter raises an
actionable error instead of inventing camera poses.

## Hypersim

Official repository: <https://github.com/apple/ml-hypersim>

The complete image release is about 1.9 TB across hundreds of 1-20 GB scene ZIPs. The downloader
uses HTTP byte ranges and extracts only selected members from each official scene archive, so a
pilot can request camera metadata, preview RGB, metric depth, and world normals without storing a
full ZIP locally.

List a pilot request without writing:

```powershell
uv run mcss download hypersim `
  --root data\hypersim_raw `
  --scenes ai_001_001 ai_001_002 ai_001_003 `
  --camera cam_00 `
  --frames 0 10 20 30 `
  --dry-run
```

Execute the selective download:

```powershell
uv run mcss download hypersim `
  --root data\hypersim_raw `
  --scenes ai_001_001 ai_001_002 ai_001_003 `
  --camera cam_00 `
  --frames 0 10 20 30
```

Prepare manifests and resized arrays:

```powershell
uv run mcss prepare hypersim `
  --raw-root data\hypersim_raw `
  --output-root data\hypersim_prepared `
  --scenes ai_001_001 ai_001_002 ai_001_003 `
  --camera cam_00 `
  --image-height 128 --image-width 160 `
  --frame-stride 5
```

### Official-split evidence-residual expansion

The expansion is derived from the local official
`data/hypersim_raw/official_metadata/metadata_images_split_scene_v1.csv`, not from a hand-written
scene list. The planner verifies its SHA-256, reconciles the old subset, and writes:

- `configs/hypersim_er_partitions.csv`: every public scene, official split, protocol partition,
  selected camera, public frame count, and prior-observation flag;
- `configs/hypersim_er_final_exclusions.csv`: the 20 test scenes already used for diagnostics and
  therefore excluded from final holdout.

Run a read-only count and storage check, then write the deterministic CSVs:

```powershell
uv run python scripts\plan_hypersim_expansion.py --dry-run
uv run python scripts\plan_hypersim_expansion.py
```

The verified plan contains 365 train, 46 validation, 20 diagnostic-test, and 26 final-holdout
scenes. Eight public scenes have no `cam_00`; the planner records a deterministic alternative
camera instead of silently dropping them. With four context views, two target views, and stride
two, the planned windows are 16,924 / 2,138 / 959 / 1,235 for those four partitions.

Download and prepare train/validation/diagnostic data with restart-safe per-scene checks:

```powershell
uv run python scripts\materialize_hypersim_expansion.py `
  --stage all `
  --partitions train val diagnostic_test `
  --workers 2 `
  --reserve-gib 100
```

Re-running the same command skips structurally complete raw and prepared scenes and retries only
incomplete work. The final holdout is sealed by default; do not add `final_holdout` or
`--allow-final-holdout` until architecture, hyperparameters, and checkpoint selection are frozen.
The 2026-08-11 empirical estimate, based on the existing 100 scenes with a 1.25 safety factor, is
about 77.6 GiB of additional raw plus prepared data.

The adapter converts the official OpenGL camera axes to this project's OpenCV convention and
records that conversion in the manifest. The default 60 degree horizontal field of view is
configurable; verify it against any camera calibration artifact supplied with your downloaded
subset before reporting final numbers.

## Storage and provenance

Keep raw archives, prepared manifests, checkpoints, and run outputs in separate directories. Each
run should record the exact scene list, camera, frame IDs, context/target counts, resize, bounds,
and config file. Never mix a rendered Replica trajectory with a mesh-only split in one result row.

Prepared manifests contain legacy scene bounds for backwards compatibility. The main A+ configs
do not use those values: localization and fixed metric support are applied when a context/target
window is loaded. Existing prepared Hypersim files therefore do not need to be regenerated for
the A+ protocol.

## Verified local snapshot

The 2026-08-10 local snapshot contains:

- Replica v1.0: 17 official archive parts in `data/replica_raw` and all 18 extracted scene assets
  in `data/replica_assets`. These are mesh/texture assets, not prepared RGB-D trajectories.
- Hypersim: `cam_00` preview RGB, metric depth, world normal, and camera metadata for a
  scene-disjoint 60 train / 20 validation / 20 test subset.
- Hypersim prepared data: 5,841 train, 1,993 validation, and 1,999 test frames below
  `data/hypersim_prepared/{train,val,test}` at 128x160.
- Exact Hypersim selection: `configs/hypersim_subset_v1.csv`, derived deterministically from the
  official `metadata_images_split_scene_v1.csv` scene split.
- Evidence-residual expansion (verified 2026-08-11): 431 raw scenes and prepared partitions with
  365 train scenes / 35,348 frames, 46 validation scenes / 4,466 frames, and 20 diagnostic-test
  scenes / 1,999 frames. The resulting 4-context/2-target/stride-2 datasets contain 16,924 / 2,138 /
  959 windows. A full idempotent rerun reported 431 complete scenes, zero downloads, zero prepares,
  and zero failures.
- The 26 final-holdout scenes / 2,576 frames remain listed in
  `configs/hypersim_er_partitions.csv` but are not present below `data/hypersim_er_prepared` or the
  raw scene root. At verification time D: retained about 265.5 GiB free.

Do not point a training config at the parent `data/hypersim_prepared` directory because recursive
manifest discovery would combine train, validation, and test scenes.
