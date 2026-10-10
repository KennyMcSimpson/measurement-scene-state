"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1: evaluation root, frame-role lock and G1 arms contract (no prediction is
made and no query depth is read here).

FRESH78 = the EVAL-V5 (23) and TRAIN-EXT (55) records in one manifest (split relabelled FRESH78, records otherwise
unchanged). Neither cohort was used by any DepthSplat run, by the reverse-JEPA training data (TRAIN72 and its
SEG-V1 segments) or by the measurement-planning audit; both are scene- and physical-scene-disjoint from TRAIN72,
DEV and the protected holdout (asserted). The arms contract seals the V21 G1 carrier (V17 L recipe, T24000) for
the three seeds, as the V33 contract did. DEV = the eight V7 DEV scenes of the DepthSplat probe.

An episode is one record and one context role (A or B): context = that role's three RGB-D frames; stream = the
Core B frame-id rule (four frames evenly spaced over the record's free frames, shared by both roles; frame ids
only); queries = primary_query. The FRESH78 streams must equal the sealed Core B V6 (TRAIN-EXT) and V7 (EVAL-V5)
stream roles. Run through cv_host_run.py from the repository root:
  python make_eval_root.py ROOT
"""

import hashlib
import json
import sys
from pathlib import Path

from mcss.mechanism_pilot.rgbd_stream_write import stream_frame_ids

REPO = Path.cwd()
COHORTS = {
    "EVAL_V5": ("outputs/EXP-3D-RGBD-EVAL-V5-DATA/manifest_eval_v5.json",
                "outputs/EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7/stream_roles.json"),
    "TRAIN_EXT": ("outputs/EXP-3D-RGBD-TRAIN-EXT-DATA/manifest_train_ext.json",
                  "outputs/EXP-3D-RGBD-DYNAMIC-TTT-CONTEXT-V6/stream_roles.json"),
}
SPLIT = "outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7/scene_split.json"
SEGMENTS = "outputs/EXP-3D-RGBD-SEGMENT-DATA-V1/manifest_seg.json"
DEV_MANIFEST = Path.home() / "cv_work/depthsplat/probe_dev/manifest_dev.json"
G1_ROOT = REPO / "outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/G1/EXP-3D-RGBD-LONG-TRAIN-V17"
SEEDS = [20260928, 20260929, 20260930]
NAME = "FRESH78"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episodes(records):
    out = []
    for r in records:
        stream = stream_frame_ids(r, "A")
        if stream is None or stream != stream_frame_ids(r, "B"):
            raise ValueError(f"{r['scene_id']}: no shared four-frame stream")
        queries = list(r["roles"]["primary_query"])
        for role, key in (("A", "context_a"), ("B", "context_b")):
            context = list(r["roles"][key])
            if set(stream) & (set(context) | set(queries)) or set(context) & set(queries):
                raise ValueError(f"{r['scene_id']}/{role}: frame roles overlap")
            out.append({"scene_id": r["scene_id"], "role": role, "context": context, "stream": list(stream),
                        "queries": queries})
    return out


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    split = json.loads((REPO / SPLIT).read_text())
    train = [r for r in split["scenes"] if r["split"] == "TRAIN"]
    dev = json.loads(DEV_MANIFEST.read_text())["scenes"]
    protected = set(split["protected_scene_ids"])
    used_ids = {r["scene_id"] for r in train} | {r["scene_id"] for r in dev} | protected
    used_physical = {r["physical_scene_id"] for r in train} | {r["physical_scene_id"] for r in dev}
    segment_bases = {r["scene_id"].rsplit("_seg", 1)[0] for r in json.loads((REPO / SEGMENTS).read_text())["scenes"]}
    if not segment_bases <= {r["scene_id"] for r in train}:
        raise ValueError("SEG-V1 holds a segment of a non-TRAIN72 scene")
    records, header, sources = [], None, {}
    for cohort, (manifest_path, stream_path) in COHORTS.items():
        m = json.loads((REPO / manifest_path).read_text())
        if header is None:
            header = {k: v for k, v in m.items() if k not in ("scenes", "schema", "candidate_lock_sha256")}
        if m["image_size"] != header["image_size"] or m["depth_semantics"] != header["depth_semantics"]:
            raise ValueError(f"{cohort}: image size or depth semantics differ")
        header["protected_scene_ids"] = sorted(set(header["protected_scene_ids"]) | set(m["protected_scene_ids"]))
        sealed_streams = json.loads((REPO / stream_path).read_text())["episodes"]
        cohort_records = [dict(r, split=NAME, source_cohort=cohort, source_manifest=manifest_path)
                          for r in m["scenes"]]
        for e in episodes(cohort_records):
            if sealed_streams.get(f"{e['scene_id']}/{e['role']}") != e["stream"]:
                raise ValueError(f"{e['scene_id']}/{e['role']}: stream differs from {stream_path}")
        if len(sealed_streams) != 2 * len(cohort_records):
            raise ValueError(f"{cohort}: sealed stream roles cover other episodes")
        records += cohort_records
        sources[cohort] = {"manifest": manifest_path, "manifest_sha256": sha(REPO / manifest_path),
                           "stream_roles": stream_path, "stream_roles_sha256": sha(REPO / stream_path),
                           "records": len(cohort_records)}
    for key in ("scene_id", "physical_scene_id"):
        values = [r[key] for r in records]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {key} across cohorts")
    if {r["scene_id"] for r in records} & used_ids or {r["physical_scene_id"] for r in records} & used_physical:
        raise ValueError("an evaluation record overlaps TRAIN72, DEV or the protected holdout")
    if {r["physical_scene_id"] for r in records} & set(header["protected_scene_ids"]):
        raise ValueError("an evaluation record is in the protected holdout")
    manifest_path = root / f"manifest_{NAME.lower()}.json"
    manifest_path.write_text(json.dumps(dict(header, scenes=records, sources=sources)))
    v17 = json.loads((REPO / "outputs/EXP-3D-RGBD-LONG-TRAIN-V17/training_contract.json").read_text())["eval"]
    contract = {
        "arms": {"G1": {"module": "mcss.mechanism_pilot.rgbd_long_train_carrier", "root": str(G1_ROOT),
                        "step": 24000, "preregistration_sha256": sha(G1_ROOT / "preregistration.json")}},
        "reference_arm": "G1",
        "seeds": SEEDS,
        "cohorts": {NAME: {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "split": NAME,
                           "role": "primary_query"}},
        "near_px": v17["near_px"],
        "reprojection_fill_constant_m": v17["reprojection_fill_constant_m"],
        "abstain_opacity_threshold": v17["abstain_opacity_threshold"],
    }
    (root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
    lock = {
        "rule": "context = role frames; stream = Core B frame-id rule (stream_frame_ids, 4 frames, shared by roles); "
                "queries = primary_query",
        "cohorts": {
            NAME: {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
                   "episodes": episodes(records)},
            "DEV": {"manifest": str(DEV_MANIFEST), "manifest_sha256": sha(DEV_MANIFEST), "episodes": episodes(dev)},
        },
        "disjoint_from": ["TRAIN72 (V7 split)", "SEG-V1 (segments of TRAIN72 scenes)", "DEV", "protected holdout"],
    }
    (root / "frame_role_lock.json").write_text(json.dumps(lock, indent=1))
    print("EVAL_ROOT", root, NAME, len(records), "records", len(lock["cohorts"][NAME]["episodes"]), "episodes; DEV",
          len(lock["cohorts"]["DEV"]["episodes"]), "episodes")


if __name__ == "__main__":
    main()
