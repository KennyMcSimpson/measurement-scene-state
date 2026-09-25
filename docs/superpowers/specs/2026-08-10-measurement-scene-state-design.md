# Measurement-Complete Scene State Design

## Research contract

The project tests whether sparse calibrated RGB observations can be compressed into one
shared scene state that supports fixed, composable, predictor-independent measurements.
The primary claim is not that target encoders are unnecessary. It is that a typed state can
be read by the same non-learned measurement program to produce RGB, ray depth, world normal,
world point map, visibility, and uncertainty, including measurement combinations withheld
during training.

This code is an experimental implementation of a promising but provisional idea. It must not
be described as CVPR-ready until matched controls and real-data experiments support it.

## Alternatives considered

1. **Dense typed voxel state (selected).** A learned context encoder writes density, color,
   uncertainty, and optional latent features into a bounded world grid. Fixed ray integration
   reads every primary output. This makes the no-bypass property inspectable and fits a 12 GB
   GPU at pilot resolution.
2. **Gaussian state.** It is efficient and easy to rasterize, but the paper would be harder to
   distinguish from amortized 3D Gaussian Splatting. It is retained as a future efficiency
   ablation, not the first implementation.
3. **Implicit SDF or tri-plane state.** It scales better than a dense grid, but a learned query
   MLP can become an undeclared target predictor. It is deferred until the explicit-state
   hypothesis survives the first falsification tests.

## Inputs and outputs

The main model consumes only context RGB, calibrated OpenCV camera matrices, fixed context-local
bounds, and target camera queries. It never receives target RGB, depth, normal, point map, or
visibility. The first preselected context camera defines the local metric frame; target labels and
full-scene manifest geometry cannot define or normalize that frame. Camera-to-world matrices use
+x right, +y down, +z forward. Depth is metric ray distance along a normalized anchor-frame ray.

The shared state contains:

- `density_logits`: one scalar per voxel;
- `color`: view-independent RGB per voxel;
- `log_variance`: aleatoric uncertainty per voxel;
- `features`: optional latent channels used only by learned-decoder controls;
- `bounds`: anchor-frame metric axis-aligned minimum and maximum.

## Architecture

Each context image passes through a compact residual 2D encoder. Every voxel center is projected
into every context camera, image features and RGB are sampled, and valid observations are fused
with mean, variance, and coverage statistics. A small 3D residual network turns those statistics
into the typed state.

The fixed renderer intersects target rays with the state bounds, samples the grid, converts
density to transmittance weights, and composes measurements. RGB is weighted color, depth is
weighted ray distance, point map is `origin + direction * depth`, visibility is accumulated
opacity, normals are finite-difference density gradients, and uncertainty is weighted variance.
The renderer has no parameters.

Two matched controls share the context encoder and state capacity:

- `learned_heads`: integrates latent features and predicts each task with separate MLP heads;
- `free_decoder`: adds query origin and direction to the integrated feature before prediction.

These controls isolate the fixed measurement interface from general model capacity. External
methods such as SRT/OSRT, pixelNeRF, pixelSplat, MVSplat, DUSt3R, MASt3R, VGGT, and per-scene 3DGS
remain separate repositories and must be rerun under a matched protocol before entering a main
comparison table.

## Data flow

`SyntheticSceneDataset` analytically renders colored spheres and boxes and is always available
for unit tests and end-to-end smoke runs. Real data is normalized to a versioned manifest with
relative file paths, intrinsics, OpenCV camera-to-world matrices, scene bounds, and modality
metadata.

Manifest bounds are a legacy/oracle field. The main A+ loader ignores them, rebases every sample
to the first context camera, and applies fixed metric local bounds. Target geometry is used only
to construct local-measurement labels after query support has been fixed from cameras and bounds.

Replica's official release contains meshes and textures rather than a ready multiview benchmark.
The downloader retrieves all official split archives with resume support. Preparation accepts a
pre-rendered Replica RGB-D trajectory or, when Habitat-Sim is installed, renders a deterministic
trajectory from the official assets. The two sources are labeled separately in run metadata.

The Hypersim downloader uses HTTP byte ranges to extract only selected scene files from official
per-scene ZIP archives. The pilot subset requests camera metadata, tone-mapped preview RGB,
metric depth, and world normals. Preparation converts the OpenGL camera orientation to the
project's OpenCV convention and scales intrinsics after resizing.

## Training and evaluation

Training supports AMP, gradient accumulation, checkpoint resume, deterministic seeds, bounded
steps, warmup-cosine or legacy constant learning rates, window/EMA telemetry, and CPU/GPU
selection. Losses are enabled only when the target modality exists. The main
losses are RGB Charbonnier, scale-aware depth L1, cosine normal, visibility BCE, point-map L1,
uncertainty NLL, density sparsity, and state total variation.

Standard metrics are PSNR, SSIM, optional LPIPS, AbsRel, RMSE, delta-1, normal mean angle and
threshold accuracy, point-map Chamfer/F-score, and visibility IoU/F1. Mechanism diagnostics are:

- measurement generalization matrix and held-out measurement gap;
- cross-context consistency;
- density/color intervention specificity;
- query-order equivariance and query-shuffle sensitivity;
- no-bypass audit proving target labels are absent from model inputs and the fixed renderer has
  zero learned parameters;
- uncertainty calibration error when uncertainty is enabled.

## Failure handling

Every downloader writes to `.part` files and atomically renames completed files. Existing files
are reused only after size checks. ZIP members are path-validated before extraction. Dataset
adapters fail with an actionable message when required camera metadata or modalities are absent.
Checkpoints contain the resolved config, model, optimizer, scaler, step, and random states.

## Verification gate

Completion requires unit tests for geometry, rendering, state construction, data manifests,
download safety, losses, metrics, and diagnostics; Ruff; a CPU synthetic forward/backward run;
and a CLI train/evaluate smoke run. Real Replica/Hypersim quality is explicitly unverified until
the datasets are downloaded and a real-data run finishes.
