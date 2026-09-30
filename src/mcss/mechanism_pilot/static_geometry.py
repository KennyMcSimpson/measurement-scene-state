"""Score-independent camera, frustum support and GT surface coverage audits.

No prediction, action, gradient, RGB, or metric ranking is used by this module.
The frozen checkpoint supplies only its geometric buffers and configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.geometry import generate_rays, project_world, transform_cameras, transform_points
from mcss.mechanism_pilot.calibration import calibrated_intrinsics, published_scene_metadata
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.types import Cameras


def summary(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    return {
        "count": int(a.size),
        "mean": float(a.mean()) if a.size else None,
        "max": float(a.max()) if a.size else None,
        "p50": float(np.quantile(a, 0.5)) if a.size else None,
        "p95": float(np.quantile(a, 0.95)) if a.size else None,
    }


def depth_to_ray_distance(depth, camera, *, semantics):
    """Explicit conversion; dataset Hypersim depth_meters already is ray distance."""
    if semantics == "ray_distance_meters":
        return depth.clone()
    if semantics != "camera_z_meters":
        raise ValueError("Unrecognized depth semantics")
    _, rays = generate_rays(camera)
    camera_rays = rays @ torch.linalg.inv(camera.c2w[:3, :3]).T
    return depth / camera_rays[..., 2]


def surface_geometry(camera, depth):
    origins, rays = generate_rays(camera)
    valid = torch.isfinite(depth) & (depth > 0)
    points = origins[valid] + rays[valid] * depth[valid, None]
    pixels, z, _ = project_world(points, camera)
    expected = valid.nonzero()[:, [1, 0]].to(pixels)
    recovered = torch.linalg.vector_norm(points - origins[valid], dim=-1)
    return (
        points,
        valid,
        pixels,
        z,
        (pixels - expected).norm(dim=-1),
        (recovered - depth[valid]).abs(),
    )


def volume_geometry(points, candidates, bounds, supported, radius):
    """Nearest candidate/supported-candidate radius is predeclared, never score tuned."""
    outside = torch.maximum(bounds[0] - points, points - bounds[1]).clamp_min(0).norm(dim=-1)
    distance = torch.cdist(points, candidates)
    nearest = distance.amin(-1)
    support_distance = (
        distance[:, supported].amin(-1)
        if supported.any()
        else torch.full_like(nearest, float("inf"))
    )
    return outside, nearest, support_distance, support_distance <= radius


def support_audit(candidates, bounds, cameras, anchor_pose):
    records, masks = [], []
    for frame_id, camera in cameras:
        anchored = transform_cameras(camera, torch.linalg.inv(anchor_pose))
        pixels, z, valid = project_world(candidates, anchored)
        finite = torch.isfinite(pixels).all(-1) & torch.isfinite(z)
        valid &= finite
        h, w = camera.image_size
        rectangle = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= w - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= h - 1)
        )
        masks.append(valid)
        records.append(
            {
                "frame_id": frame_id,
                "projected_uv": pixels.tolist(),
                "camera_z": z.tolist(),
                "in_front": (z > torch.finfo(z.dtype).eps).tolist(),
                "inside_image_rectangle": rectangle.tolist(),
                "finite": finite.tolist(),
                "sampled_feature_valid": valid.tolist(),
                "failure_counts": {
                    "nonfinite": int((~finite).sum()),
                    "behind_or_zero_z": int((z <= torch.finfo(z.dtype).eps).sum()),
                    "front_but_outside_image": int(
                        ((z > torch.finfo(z.dtype).eps) & ~rectangle & finite).sum()
                    ),
                },
            }
        )
    counts = torch.stack(masks).sum(0)
    used = counts >= 2
    return {
        "candidate_anchor_xyz": candidates.tolist(),
        "candidate_world_xyz": transform_points(candidates, anchor_pose).tolist(),
        "within_volume": ((candidates >= bounds[0]) & (candidates <= bounds[1])).all(-1).tolist(),
        "cameras": records,
        "support_count": counts.tolist(),
        "used_by_fuse": used.tolist(),
        "used_by_complete": used.tolist(),
        "at_least_one": int((counts >= 1).sum()),
        "at_least_two": int(used.sum()),
        "histogram": {str(i): int((counts == i).sum()) for i in range(len(cameras) + 1)},
        "spatial_distribution_supported_anchor_xyz": candidates[used].tolist(),
    }, used


def audit_manifest(
    manifest_path,
    checkpoint_path,
    output_dir,
    *,
    primary_frames=(8, 9),
    secondary_frames=(12, 13, 14, 15),
    seed=20260927,
    metadata_csv=None,
):
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("depth_semantics") != "ray_distance_meters":
        raise ValueError("Audit requires explicit metric ray-distance manifest")
    carrier, _, checkpoint = load_dynamic_checkpoint(checkpoint_path)
    candidates, bounds = carrier._candidate_points.detach(), carrier._bounds.detach().reshape(2, 3)
    grid = carrier.config.grid_size
    voxel = (bounds[1] - bounds[0]) / torch.tensor([grid[2], grid[1], grid[0]])
    radius = float(voxel.norm() / 2)
    metadata = published_scene_metadata(metadata_csv) if metadata_csv else {}
    camera_rows, coverage_rows, support_rows, access, sampled_rows = [], [], {}, [], []
    rng = np.random.default_rng(seed)
    for scene in manifest["scenes"]:
        sid = scene["scene_id"]
        frames = {f["frame_id"]: f for f in scene["frames"]}
        size = tuple(manifest["image_size"])
        cameras = {
            i: Cameras(
                torch.tensor(f["intrinsics"], dtype=torch.float32),
                torch.tensor(f["c2w"], dtype=torch.float32),
                size,
            )
            for i, f in frames.items()
        }
        anchor_id = scene["roles"]["context_a"][0]
        anchor = cameras[anchor_id].c2w
        groups = {k: scene["roles"].get(k, []) for k in ("context_a", "context_b", "stream")}
        groups["anchor"] = [anchor_id]
        groups["context_a_plus_stream"] = sorted(set(groups["context_a"] + groups["stream"]))
        scene_support, supported = {}, {}
        for role, ids in groups.items():
            if not ids:
                continue
            scene_support[role], supported[role] = support_audit(
                candidates, bounds, [(i, cameras[i]) for i in ids], anchor
            )
        scene_support["A_B_supported_intersection"] = int(
            (supported["context_a"] & supported["context_b"]).sum()
        )
        scene_support["definition"] = (
            "Geometric finite in-frustum sampling eligibility; not occlusion visi"
            "bility or evidence of a surface. Feature values are not inspected. B"
            "oth branches require >=2 views."
        )
        support_rows[sid] = {
            r: {
                k: v
                for k, v in record.items()
                if k in ("at_least_one", "at_least_two", "histogram")
            }
            for r, record in scene_support.items()
            if isinstance(record, dict)
        }
        write_json(output_dir / f"support_audit_{sid}.json", scene_support)
        for frame_id, frame in frames.items():
            path = Path(frame["depth"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            access.append(
                {
                    "scene_id": sid,
                    "frame_id": frame_id,
                    "kind": "GT_depth_camera",
                    "purpose": "static_geometry_audit_not_model_selection",
                    "path": str(path),
                    "sha256": sha(path),
                }
            )
            depth = torch.tensor(np.load(path).squeeze(), dtype=torch.float64)
            camera = cameras[frame_id].to(dtype=torch.float64)
            points, valid, pixels, z, pe, de = surface_geometry(camera, depth)
            anchored_points = transform_points(points, torch.linalg.inv(anchor.double()))
            outside, nearest, _, _ = volume_geometry(
                anchored_points,
                candidates.double(),
                bounds.double(),
                supported["context_a"],
                radius,
            )
            relative = torch.linalg.inv(anchor.double()) @ camera.c2w
            row = {
                "scene_id": sid,
                "frame_id": frame_id,
                "roles": [k for k, ids in groups.items() if frame_id in ids],
                "primary": frame_id in scene["roles"].get("primary_query", primary_frames),
                "secondary": frame_id in scene["roles"].get("secondary_query", secondary_frames),
                "valid_GT_fraction": float(valid.double().mean()),
                "depth_m": summary(depth[valid].numpy()),
                "inside_volume_fraction": float((outside == 0).double().mean()),
                "outside_distance_m": summary(outside.numpy()),
                "nearest_candidate_distance_m": summary(nearest.numpy()),
                "any_candidate_neighborhood_fraction": float((nearest <= radius).double().mean()),
                "anchor_camera_baseline_m": float(relative[:3, 3].norm()),
                "anchor_relative_forward": relative[:3, 2].tolist(),
                "context_supported_neighborhood": {},
            }
            arrays = {
                "valid_pixel_yx": valid.nonzero().numpy(),
                "surface_anchor_xyz": anchored_points.numpy(),
                "outside_distance_m": outside.numpy(),
                "nearest_candidate_distance_m": nearest.numpy(),
            }
            for role, mask in supported.items():
                _, _, ds, covered = volume_geometry(
                    anchored_points, candidates.double(), bounds.double(), mask, radius
                )
                row["context_supported_neighborhood"][role] = {
                    "surface_support_fraction": float(covered.double().mean()),
                    "no_candidate_neighborhood_support_fraction": float((~covered).double().mean()),
                    "nearest_supported_distance_m": summary(ds.numpy()),
                }
                arrays[f"{role}_nearest_supported_m"] = ds.numpy()
            np.savez_compressed(output_dir / f"{sid}_{frame_id:04d}_geometry_raw.npz", **arrays)
            coverage_rows.append(row)
            camera_rows.append(
                {
                    "scene_id": sid,
                    "frame_id": frame_id,
                    "projection_roundtrip_mean": float(pe.mean()),
                    "projection_roundtrip_max": float(pe.max()),
                    "depth_roundtrip_mean": float(de.mean()),
                    "depth_roundtrip_max": float(de.max()),
                    "pose_rotation_orthogonality_max": float(
                        (camera.c2w[:3, :3].T @ camera.c2w[:3, :3] - torch.eye(3)).abs().max()
                    ),
                    "camera_z_to_ray_ratio": summary((z / depth[valid]).numpy()),
                }
            )
            origins, rays = generate_rays(camera)
            valid_yx = valid.nonzero()
            for ix in rng.choice(len(points), size=min(8, len(points)), replace=False):
                y, x = valid_yx[ix].tolist()
                sampled_rows.append(
                    {
                        "scene_id": sid,
                        "frame_id": frame_id,
                        "pixel_xy": [x, y],
                        "camera_ray_world": rays[y, x].tolist(),
                        "GT_ray_distance_m": float(depth[y, x]),
                        "world_point": points[ix].tolist(),
                        "reprojected_xy": pixels[ix].tolist(),
                        "camera_z_m": float(z[ix]),
                        "renderer_ray_parameter_t_m": float(depth[y, x]),
                    }
                )
        # Independent native GT position validation, when downloaded; never model prediction.
        raw = manifest_path.parent / "raw" / sid / "images" / "scene_cam_00_geometry_hdf5"
        position_file = raw / f"frame.{anchor_id:04d}.position.hdf5"
        depth_file = raw / f"frame.{anchor_id:04d}.depth_meters.hdf5"
        if sid in metadata and position_file.exists() and depth_file.exists():
            for native_path in (position_file, depth_file):
                access.append(
                    {
                        "scene_id": sid,
                        "frame_id": anchor_id,
                        "kind": "native_GT_position_or_depth",
                        "purpose": "independent_depth_units_audit",
                        "path": str(native_path),
                        "sha256": sha(native_path),
                    }
                )
            with h5py.File(position_file) as f:
                position = np.array(f["dataset"]) * float(
                    metadata[sid]["settings_units_info_meters_scale"]
                )
            with h5py.File(depth_file) as f:
                native_depth = torch.tensor(np.array(f["dataset"]), dtype=torch.float64)
            native_k = torch.tensor(
                calibrated_intrinsics(metadata[sid], native_depth.shape), dtype=torch.float64
            )
            native_cam = Cameras(native_k, anchor.double(), tuple(native_depth.shape))
            orig, rays = generate_rays(native_cam)
            expected = orig + rays * native_depth[..., None]
            good = (
                np.isfinite(position).all(-1)
                & np.isfinite(native_depth.numpy())
                & (native_depth.numpy() > 0)
            )
            error = np.linalg.norm(expected.numpy()[good] - position[good], axis=-1)
            resized_k = torch.tensor(
                calibrated_intrinsics(metadata[sid], size), dtype=torch.float64
            )
            result = {
                "scene_id": sid,
                "native_position_error_m": summary(error),
                "resize_intrinsics_max_error": float(
                    (resized_k - cameras[anchor_id].intrinsics.double()).abs().max()
                ),
                "position_units_multiplier": float(
                    metadata[sid]["settings_units_info_meters_scale"]
                ),
                "GT_depth_conversion": "identity already meters",
                "position_file_sha256": sha(position_file),
                "depth_file_sha256": sha(depth_file),
            }
        else:
            result = {
                "scene_id": sid,
                "native_position_check": "UNAVAILABLE",
                "reason": "Native depth/position or published metadata absent",
            }
        write_json(output_dir / f"native_geometry_{sid}.json", result)
    write_json(output_dir / "support_audit.json", support_rows)
    write_json(output_dir / "camera_geometry_audit.json", camera_rows)
    write_json(
        output_dir / "volume_coverage.json",
        {
            "neighborhood_rule": (
                "one voxel HALF diagonal, fixed before metrics; "
                "centers eligible at >=2 view frustum support"
            ),
            "radius_m": radius,
            "voxel_xyz_m": voxel.tolist(),
            "frames": coverage_rows,
        },
    )
    write_json(output_dir / "sampled_GT_pixels.json", sampled_rows)
    (output_dir / "geometry_gt_access.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in access)
    )
    report = {
        "checkpoint_sha256": checkpoint["sha256"],
        "manifest_sha256": sha(manifest_path),
        "geometry_frame": (
            "anchor camera coordinates for volume; "
            "explicit world coordinates retained for candidate audit"
        ),
        "projection_roundtrip_max": max(r["projection_roundtrip_max"] for r in camera_rows),
        "depth_roundtrip_max": max(r["depth_roundtrip_max"] for r in camera_rows),
        "PREVIOUS_DEPTH_METRIC_INVALID": False,
        "semantics": (
            "GT and renderer use metric normalized-ray parameter. "
            "Renderer depth is UNCONDITIONAL sum(weights*t), not opacity-normalized; "
            "low opacity is a measurement bottleneck, not ray/z conversion bug."
        ),
        "model_predictions_read": False,
        "DYNAMIC_TTT_RUN": False,
        "seed": seed,
    }
    write_json(output_dir / "geometry_audit.json", report)
    (output_dir / "depth_definition.md").write_text(
        "# Depth definition\n\nHypersim `depth_meters` is distance along the un"
        "it camera ray in meters. The stored array is used unchanged; asset-u"
        "nit scale applies to camera translations and native position, not to"
        " depth_meters. OpenGL camera axes are converted to OpenCV x-right/y-"
        "down/z-forward in data preparation. Per-scene published inverse proj"
        "ection and integer pixel centers with half-pixel resize define K.\n\nG"
        "T world point = camera origin + normalize(R K^-1 [u,v,1]) * depth. C"
        "amera-z is ray distance times camera-ray z and is generally differen"
        "t. The renderer uses normalized rays and metric samples t, but retur"
        "ns sum(w*t), with zero contribution from escaped mass. It does not d"
        "ivide by opacity. Comparing this unconditional estimate against GT r"
        "ay distance retains the frozen metric and exposes opacity/coverage f"
        "ailure; no hidden rescaling or depth semantics change is made. Nativ"
        "e GT position and dataset camera round-trips are checked independent"
        "ly from any model prediction.\n"
    )
    return report
