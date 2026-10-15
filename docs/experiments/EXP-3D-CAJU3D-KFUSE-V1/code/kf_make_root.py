"""EXP-3D-CAJU3D-KFUSE-V1: the root files, from metadata only (no image, depth or prediction is read).

Cohorts (episodes copied byte for byte from the source frame-role locks; place = physical_scene_id):
  FIT_SYSTEM / FIT_SYSTEM2   every ARKIT_SYSTEM (FULL-SYSTEM V1) and ARKIT_SYSTEM2 (V2) episode: the fusion tables
                             are fitted here (their truth has been read by earlier experiments)
  CONF_SYSTEM3               every ARKIT_SYSTEM3 episode (V3): P1
  CONF_VAL                   every ARKIT_VAL episode (ENSEMBLES-ARKIT): P2
The cohorts share no place, and none of their scans or places is in the reverse H-JEPA's training data
(EXP-3D-RGBD-ARKIT-DATA-V1 TRAIN and its DEV monitor).
reused_seals.json lists per cohort the sealed geometry (R7, near7, cosine) and the released-weights predictions
(TTT_m1 = Core B's TTT) with their hashes.
Run through cv_host_run.py from the repository root (CUDA hidden):  python kf_make_root.py ROOT
"""

import hashlib
import json
import sys
from pathlib import Path

OUT = Path("outputs").resolve()
COHORTS = {"FIT_SYSTEM": ("EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1", "ARKIT_SYSTEM"),
           "FIT_SYSTEM2": ("EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2", "ARKIT_SYSTEM2"),
           "CONF_SYSTEM3": ("EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3", "ARKIT_SYSTEM3"),
           "CONF_VAL": ("EXP-3D-RGBD-ENSEMBLES-ARKIT-V1", "ARKIT_VAL")}
JEPA_DATA = OUT / "EXP-3D-RGBD-ARKIT-DATA-V1/manifest_arkit_trainval.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    if path.exists():
        raise SystemExit(f"{path} exists already")
    path.write_text(json.dumps(value, indent=1) + "\n")


def seal_dir(directory):
    parts = sorted((OUT / directory).glob("ds_seal_*of*.json"))
    if not parts:
        raise FileNotFoundError(directory)
    return {"dir": str(OUT / directory), "parts": {p.name: sha(p) for p in parts}}


def main():
    root = Path(sys.argv[1]).resolve()
    cohorts, reused = {}, {}
    for name, (exp, source) in COHORTS.items():
        lock = json.loads((OUT / exp / "frame_role_lock.json").read_text())["cohorts"][source]
        manifest = Path(lock["manifest"])
        if sha(manifest) != lock["manifest_sha256"]:
            raise PermissionError(f"{exp}: manifest changed since its lock")
        place = {r["scene_id"]: r["physical_scene_id"] for r in json.loads(manifest.read_text())["scenes"]}
        episodes = [dict(e, place=place[e["scene_id"]]) for e in lock["episodes"]]
        cohorts[name] = {"source": f"{exp}:{source}", "manifest": str(manifest), "manifest_sha256": sha(manifest),
                         "episodes": episodes}
        geometry = OUT / exp / "geometry" / f"geometry_seal_{source}.json"
        reused[name] = {"geometry": {"path": str(geometry), "sha256": sha(geometry)},
                        "ZS": seal_dir(f"{exp}/ds/{source}/ZS")}
    places = {n: {e["place"] for e in c["episodes"]} for n, c in cohorts.items()}
    scans = {n: {e["scene_id"] for e in c["episodes"]} for n, c in cohorts.items()}
    names = list(places)
    for a in names:
        for b in names[names.index(a) + 1:]:
            if places[a] & places[b]:
                raise ValueError(f"{a} and {b} share a place")
    jepa = [r for r in json.loads(JEPA_DATA.read_text())["scenes"] if r["split"] in ("TRAIN", "DEV")]
    jepa_places = {r["physical_scene_id"] for r in jepa}
    jepa_scans = {r["scene_id"] for r in jepa}
    for n in names:
        if places[n] & jepa_places or scans[n] & jepa_scans:
            raise ValueError(f"{n} overlaps the H-JEPA's training data")
    root.mkdir(parents=True, exist_ok=True)
    write_new(root / "frame_role_lock.json", {
        "rule": "every episode of the source frame-role locks (context = role frames; stream = Core B frame-id rule; "
                "queries = primary_query); place = physical_scene_id",
        "jepa_training_data": {"manifest": str(JEPA_DATA), "sha256": sha(JEPA_DATA), "splits": ["TRAIN", "DEV"],
                               "places": len(jepa_places), "overlap_with_cohorts": 0},
        "cohorts": cohorts})
    write_new(root / "reused_seals.json", reused)
    print("KF_ROOT", {n: (len(c["episodes"]), len(places[n])) for n, c in cohorts.items()})


if __name__ == "__main__":
    main()
