"""EXP-3D-CAJU3D-ENGINE-V1: the root files, from metadata only (no image, depth or prediction is read).

Cohorts (episodes copied byte for byte from the source frame-role locks; place = physical_scene_id):
  DEV_SYSTEM / DEV_SYSTEM2   12 seeded scans each of ARKIT_SYSTEM (FULL-SYSTEM V1) and ARKIT_SYSTEM2 (V2)
  CONF_SYSTEM3               every ARKIT_SYSTEM3 episode (V3), the primary confirmation cohort
  CONF_VAL                   every ARKIT_VAL episode (ENSEMBLES-ARKIT), the replication cohort
None of these places was seen by JN1 in training (JN1 trained on the ARKit TRAIN places only).
reused_seals.json lists, per cohort, the sealed geometry (R7), released-weights CACHE / TTT_m1 (Core B) and JN1 CACHE
predictions this experiment reads, with their hashes.
Run through cv_host_run.py from the repository root (CUDA hidden):  python engine_make_root.py ROOT
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_logic as L  # noqa: E402

OUT = Path("outputs").resolve()
JN1_CACHE = {"DEV_SYSTEM": "EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1/ds/ARKIT_SYSTEM/JN1",
             "DEV_SYSTEM2": "EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1/ds/ARKIT_SYSTEM2/JN1",
             "CONF_SYSTEM3": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1/ds/ARKIT_SYSTEM3/JN1",
             "CONF_VAL": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1/ds/ARKIT_VAL/JN"}


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
    sources = {**{k: v for k, v in L.DEV_COHORTS.items()}, **{k: v for k, v in L.CONF_COHORTS.items()}}
    for i, (name, (exp, source)) in enumerate(sources.items()):
        lock = json.loads((OUT / exp / "frame_role_lock.json").read_text())["cohorts"][source]
        manifest = Path(lock["manifest"])
        if sha(manifest) != lock["manifest_sha256"]:
            raise PermissionError(f"{exp}: manifest changed since its lock")
        place = {r["scene_id"]: r["physical_scene_id"] for r in json.loads(manifest.read_text())["scenes"]}
        episodes = lock["episodes"]
        if name in L.DEV_COHORTS:
            keep = set(L.select_dev_scans([e["scene_id"] for e in episodes], i))
            episodes = [e for e in episodes if e["scene_id"] in keep]
        episodes = [dict(e, place=place[e["scene_id"]]) for e in episodes]
        cohorts[name] = {"source": f"{exp}:{source}", "manifest": str(manifest), "manifest_sha256": sha(manifest),
                         "episodes": episodes}
        reused[name] = {"geometry": {"path": str(OUT / exp / "geometry" / f"geometry_seal_{source}.json"),
                                     "sha256": sha(OUT / exp / "geometry" / f"geometry_seal_{source}.json")},
                        "ZS": seal_dir(f"{exp}/ds/{source}/ZS"), "JN1_CACHE": seal_dir(JN1_CACHE[name])}
    places = {n: {e["place"] for e in c["episodes"]} for n, c in cohorts.items()}
    names = list(places)
    for a in names:
        for b in names[names.index(a) + 1:]:
            if places[a] & places[b]:
                raise ValueError(f"{a} and {b} share a place")
    root.mkdir(parents=True, exist_ok=True)
    write_new(root / "frame_role_lock.json", {
        "rule": "episodes copied from the source frame-role locks (context = role frames; stream = Core B frame-id "
                "rule; queries = primary_query); DEV = 12 seeded scans per source (engine_logic.select_dev_scans)",
        "cohorts": cohorts})
    write_new(root / "reused_seals.json", reused)
    print("ENGINE_ROOT", {n: (len(c["episodes"]), len(places[n])) for n, c in cohorts.items()})


if __name__ == "__main__":
    main()
