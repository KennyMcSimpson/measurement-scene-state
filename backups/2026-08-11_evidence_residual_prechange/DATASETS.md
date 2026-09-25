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

Do not point a training config at the parent `data/hypersim_prepared` directory because recursive
manifest discovery would combine train, validation, and test scenes.
