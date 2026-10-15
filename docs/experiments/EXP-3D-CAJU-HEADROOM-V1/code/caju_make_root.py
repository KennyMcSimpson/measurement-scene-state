"""EXP-3D-CAJU-HEADROOM-V1: the root files, from metadata only (frame ids, roles, place ids; no image, depth or model
output is read).

Cohorts:
  CAJU_PILOT   the 8 DEV records of EXP-3D-RGBD-ARKIT-DATA-V1 (engineering pilot; never in the formal statistics)
  CAJU_SEL     16 TRAIN places (seeded split, caju_logic.select_places): the best fixed action is chosen here
  CAJU_EVAL    34 TRAIN places: the oracle headroom is measured here
  CAJU_COMPAT  the first two ARKIT_SYSTEM3 records of EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3 (engine compatibility only:
               this experiment's OFF / FULL against V3's sealed CACHE / TTT_m1, no ground truth read)
The other 132 TRAIN places stay untouched. Episodes follow V3's rule (caju_logic.build_episodes).
Writes frame_role_lock.json, scene_manifest.json, episode_manifest.json, action_contract.json and
information_contract.json; refuses to overwrite any of them.
Run through cv_host_run.py from the repository root (CUDA hidden):  python caju_make_root.py ROOT
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caju_logic import (  # noqa: E402
    ACTIONS,
    ALPHA_MIN,
    STEPS,
    build_episodes,
    select_places,
)
from mcss.mechanism_pilot.rgbd_stream_write import stream_frame_ids  # noqa: E402

TRAINVAL = Path("outputs/EXP-3D-RGBD-ARKIT-DATA-V1/manifest_arkit_trainval.json").resolve()
V3 = Path("outputs/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3").resolve()
V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()
OUTPUTS = ("frame_role_lock.json", "scene_manifest.json", "episode_manifest.json", "action_contract.json",
           "information_contract.json")
# Prior use of the TRAIN / DEV records (from the experiment records; none ran Core B on them).
EXPOSURE = {
    "TRAIN": "training data of the reverse-JEPA DepthSplat variants (JA, JC, JN, JK, JO; JL / JOL inside the 502-scan "
             "set) and of the GA carrier (151 records); never evaluated with Core B (released weights, CACHE / TTT)",
    "DEV": "development monitor of the reverse-JEPA trainings (released and fine-tuned CTX arm, cap 3.0, primary-query "
           "depth logged); never evaluated with Core B",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    if path.exists():
        raise SystemExit(f"{path} exists already")
    path.write_text(json.dumps(value, indent=1) + "\n")


def episodes_of(records):
    out = []
    for r in records:
        stream = stream_frame_ids(r, "A")
        if stream is None or stream != stream_frame_ids(r, "B"):
            raise ValueError(f"{r['scene_id']}: no shared four-frame stream")
        out.extend(build_episodes(r, stream))
    return out


def main():
    root = Path(sys.argv[1]).resolve()
    if any((root / name).exists() for name in OUTPUTS):
        raise SystemExit("root files exist already")
    manifest = json.loads(TRAINVAL.read_text())
    if manifest["protected_scene_ids"]:
        raise ValueError("the ARKit manifest lists protected scenes; this design assumes none")
    records = manifest["scenes"]
    train = [r for r in records if r["split"] == "TRAIN"]
    dev = [r for r in records if r["split"] == "DEV"]
    if len({r["physical_scene_id"] for r in records}) != len(records):
        raise ValueError("a place holds more than one record")
    split = select_places([r["physical_scene_id"] for r in train])
    by_place = {r["physical_scene_id"]: r for r in train}
    v3_lock = json.loads((V3 / "frame_role_lock.json").read_text())["cohorts"]["ARKIT_SYSTEM3"]
    v3_manifest = Path(v3_lock["manifest"])
    if sha(v3_manifest) != v3_lock["manifest_sha256"]:
        raise PermissionError("V3 manifest changed since its lock")
    v3_records = {r["scene_id"]: r for r in json.loads(v3_manifest.read_text())["scenes"]}
    compat_ids = list(dict.fromkeys(e["scene_id"] for e in v3_lock["episodes"]))[:2]
    compat = episodes_of([v3_records[s] for s in compat_ids])
    for e in compat:
        twin = next(x for x in v3_lock["episodes"] if x["scene_id"] == e["scene_id"] and x["role"] == e["role"])
        if any(twin[k] != e[k] for k in ("context", "stream", "queries")):
            raise ValueError(f"{e['scene_id']}/{e['role']}: rule differs from V3's lock")
    cohorts = {
        "CAJU_PILOT": (TRAINVAL, episodes_of(dev)),
        "CAJU_SEL": (TRAINVAL, episodes_of([by_place[p] for p in split["CAJU_SEL"]])),
        "CAJU_EVAL": (TRAINVAL, episodes_of([by_place[p] for p in split["CAJU_EVAL"]])),
        "CAJU_COMPAT": (v3_manifest, compat),
    }
    places = {name: sorted({e["place"] for e in eps}) for name, (_, eps) in cohorts.items()}
    formal = set(places["CAJU_SEL"]) | set(places["CAJU_EVAL"])
    if set(places["CAJU_SEL"]) & set(places["CAJU_EVAL"]) or formal & set(places["CAJU_PILOT"]):
        raise ValueError("cohorts share a place")
    rule = ("context = role frames; stream = stream_frame_ids (4 evenly spaced free frames, shared by roles); "
            "queries = primary_query (after every arrived frame); history = the free frames the stream leaves out "
            "(never measured, inside the stream span)")
    root.mkdir(parents=True, exist_ok=True)
    write_new(root / "frame_role_lock.json", {"rule": rule, "cohorts": {
        name: {"manifest": str(m), "manifest_sha256": sha(m), "episodes": eps} for name, (m, eps) in cohorts.items()}})
    write_new(root / "scene_manifest.json", {
        "unit": "physical place (ARKitScenes venue, physical_scene_id); one scan per place in every cohort",
        "selection": {"function": "caju_logic.select_places", "seed": 20261014, "pool": "the 182 TRAIN places",
                      "sizes": {"CAJU_SEL": 16, "CAJU_EVAL": 34}},
        "places": places, "reserved_untouched_train_places": split["RESERVED_UNTOUCHED"],
        "scans": {name: sorted({e["scene_id"] for e in eps}) for name, (_, eps) in cohorts.items()},
        "exposure": {"CAJU_SEL": EXPOSURE["TRAIN"], "CAJU_EVAL": EXPOSURE["TRAIN"], "CAJU_PILOT": EXPOSURE["DEV"],
                     "CAJU_COMPAT": "ARKIT_SYSTEM3, the confirmation cohort of V3, NF, NS, AT, CA and CW (reused here "
                                    "only to check the engine against V3's seals)",
                     "released_depthsplat": "trained on RealEstate10K + DL3DV; no ARKitScenes exposure"},
        "fresh_test_claim": False,
        "final_holdout_touched": False,
        "manifest_basis": manifest["scenes"][0]["historical_exposure_basis"],
    })
    write_new(root / "episode_manifest.json", {name: [dict(e, cohort=name) for e in eps]
                                               for name, (_, eps) in cohorts.items()})
    write_new(root / "action_contract.json", {
        "start_state": "CACHE: released DepthSplat (re10k+dl3dv base, Gaussian scale cap 0.1) on the 3 context + 4 "
                       "stream RGB-D views, measured planar depth injected as the final context depth",
        "actions": {a: {"ttt_steps": STEPS[a]} for a in ACTIONS},
        "ttt": {"engine": "V1 ds_eval.ttt (EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/ds_eval.py), unchanged math",
                "optimized": ["means", "per-axis log-scales in each Gaussian's eigenbasis (rotations fixed)",
                              "opacity logits", "SH DC", "higher SH"],
                "network_weights": "frozen", "optimizer": "Adam, eps 1e-15, fresh per episode",
                "lr": "3DGS defaults x 1 (means 1.6e-4 x median measured planar depth, log-scales 5e-3, opacity "
                      "logits 2.5e-2, SH DC 2.5e-3, higher SH 1.25e-4)",
                "loss": "relative L1 of the raw rendered depth on valid measured pixels + 0.5 x RGB L1, on the 7 "
                        "arrived frames",
                "failure": "a non-finite loss reverts the state to CACHE (as V1)",
                "branching": "SHORT is the state after step 10 of the FULL trajectory, read out before the trajectory "
                             "continues; the pilot checks it against an independent 10-step run"},
        "readout": {"ray": "depth_z / alpha / cosine", "F7": f"ray where alpha >= {ALPHA_MIN}, else R7",
                    "CORE_B": "sqrt(R7 x F7), both floored at 1e-6", "R7": "REPROJ7_HARMONIC of the 7 arrived frames"},
        "cost": {"unit": "TTT optimizer steps (OFF 0, SHORT 10, FULL 100)",
                 "measured": "CUDA-synchronized seconds per segment, reported but not used in any gate"},
        "v1_ds_eval_sha256": sha(V1 / "code/ds_eval.py"),
    })
    write_new(root / "information_contract.json", {
        "replay": "reads the RGB, measured depth and camera of the 7 arrived frames only; query and history frames "
                  "contribute their cameras (metadata) to render predictions; every opened file is audited per "
                  "episode and must belong to an arrived frame",
        "pre_update_features": "computed from the CACHE state rendered at the 7 arrived frames and their measured "
                               "RGB-D only; no query or history camera, image or depth",
        "future_query_camera_to_controller": False,
        "geometry": "R7 from the measured depth of the 7 arrived frames (teacher path, purpose logged) at the query "
                    "and history cameras",
        "scoring": "query and history depth are read only after every seal of the cohort is complete; the fixed "
                   "action is chosen on CAJU_SEL and its record hashed before any CAJU_EVAL depth is read",
        "final_holdout_touched": False,
    })
    print("CAJU_ROOT", root, {name: len(eps) for name, (_, eps) in cohorts.items()}, "episodes")


if __name__ == "__main__":
    main()
