#!/usr/bin/env python3
"""V20 P2: the width contrast with V18 at step 24000, the width x length 2x2 and the branch.

WIDTH_GAIN_24K = FAR AbsRel(V18 L at T24000) - FAR AbsRel(V20 L at T24000) over the pooled
EVAL-V3 + EVAL-V4 scenes, paired by scene: both experiments score the same frozen cohorts and
queries, and each seed sees the same data in both. V11's frozen aggregation, scene bootstrap and V2
gate are used, as for P1. Before V18's rows are read, the frozen V18 is checked: its preregistration
hash, its finished analysis with its own built-in control PASS, and its scored rows unchanged since
that analysis. If V18's built-in control failed, WIDTH_STATUS is NO_DATA and the branch follows P1
alone. V18 started earlier and is expected to finish first; otherwise this waits for it.
The descriptive cohorts (EVAL-FAR, FRESH-V2) get the same contrasts, without a gate.
Writes width_contrast_results.json with the final INTERPRETATION_BRANCH.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import time
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_lock
from mcss.mechanism_pilot.small_training import sha, write_json

PRIMARY_COHORTS = ("EVAL_V3", "EVAL_V4")
DESCRIPTIVE_COHORTS = ("EVAL_FAR", "FRESH_V2")
SCRIPTS = Path(__file__).resolve().parent
POLL_SECONDS = 600
MAX_WAIT_SECONDS = 48 * 3600
DEFINITIONS = {
    "WIDTH_GAIN_24K": "FAR AbsRel(V18 L at T24000) - FAR AbsRel(V20 L at T24000)",
    "WIDTH_GAIN_6K": "FAR AbsRel(V18 L at T6000) - FAR AbsRel(V20 L at T6000)",
    "INTERACTION": "[FAR(V18 T6000) - FAR(V18 T24000)] - [FAR(V20 T6000) - FAR(V20 T24000)]; "
    "negative means the wider carrier gains more from longer training",
}


@functools.cache
def host():
    """V11's frozen EVAL-V3 module: aggregation, bootstrap and V2 gate."""
    spec = importlib.util.spec_from_file_location(
        "v20_width_host_v11_eval", SCRIPTS / "evaluate_rgbd_completion_eval_v3.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def width_contrast(rows17, rows19, *, gate):
    """WIDTH_GAIN_24K, WIDTH_GAIN_6K, INTERACTION and the four absolute FAR values."""
    v11 = host()
    values = {}
    for width, rows in (("W32", rows17), ("W64", rows19)):
        for step in ("T6000", "T24000"):
            values[width, step] = v11.scene_values(rows, step, "absrel_FAR")
    scenes = [set(v) for v in values.values()]
    if any(s != scenes[0] for s in scenes):
        raise PermissionError("V18 and V20 must cover the same scenes with FAR pixels")
    gain32 = {s: values["W32", "T6000"][s] - values["W32", "T24000"][s] for s in scenes[0]}
    gain64 = {s: values["W64", "T6000"][s] - values["W64", "T24000"][s] for s in scenes[0]}
    contrasts = {
        "WIDTH_GAIN_24K": v11.contrast(values["W32", "T24000"], values["W64", "T24000"]),
        "WIDTH_GAIN_6K": v11.contrast(values["W32", "T6000"], values["W64", "T6000"]),
        "INTERACTION": v11.contrast(gain32, gain64),
    }
    out = {
        "contrasts": {
            name: {"gain": stats, "definition": DEFINITIONS[name]}
            for name, stats in contrasts.items()
        },
        "absolute_far": {f"{w}_{s}": v11.summary(v) for (w, s), v in values.items()},
        "length_gain_far": {"W32": v11.summary(gain32), "W64": v11.summary(gain64)},
        "n_scenes": len(scenes[0]),
    }
    item = out["contrasts"]["WIDTH_GAIN_24K"]
    if gate and item["gain"] is not None:
        item["gate"], item["gate_checks"] = v11.gate(item["gain"])
    return out


def branch(length_status, width_status):
    first, second = length_status == "SUPPORTED", width_status == "SUPPORTED"
    return "A" if first and second else "B" if first or second else "C"


def v18_rows(directory, preregistration_sha256, *, poll=POLL_SECONDS, max_wait=MAX_WAIT_SECONDS):
    """The frozen V18's scored query rows, or None when its built-in control failed."""
    directory = Path(directory)
    if sha(directory / "preregistration.json") != preregistration_sha256:
        raise PermissionError("V18 is not the frozen V18")
    waited = 0
    while True:
        control = directory / "audit/v14_prefix_control.json"
        if control.exists() and read(control)["status"] != "PASS":
            return None, {"state": "CONTROL_FAILED"}
        result = directory / "eval_results.json"
        if result.exists():
            break
        if waited >= max_wait:
            raise TimeoutError("V18 did not finish its analysis")
        time.sleep(poll)
        waited += poll
    analysis = read(result)
    rows_path = directory / "raw/eval/query_rows.json"
    if analysis["control"] != "PASS" or list(analysis["input_sha256"].values()) != [sha(rows_path)]:
        raise PermissionError("V18's scored rows changed after its analysis")
    record = {
        "state": "COMPLETE",
        "eval_results_sha256": sha(result),
        "query_rows_sha256": sha(rows_path),
        "length_status": analysis["statuses"]["LENGTH_STATUS"],
        "branch": analysis["INTERPRETATION_BRANCH"],
    }
    return read(rows_path), record


def run(root):
    root = Path(root).resolve()
    _, config = verify_lock(root)
    frozen = config["width_contrast"]
    analysis = read(root / "eval_results.json")
    rows_path = root / "raw/eval/query_rows.json"
    if list(analysis["input_sha256"].values()) != [sha(rows_path)]:
        raise PermissionError("V20's scored rows changed after its analysis")
    rows19 = read(rows_path)
    rows17, v18 = v18_rows(frozen["directory"], frozen["preregistration_sha256"])
    length_status = analysis["statuses"]["LENGTH_STATUS"]
    blocks = {}
    if rows17 is None:
        width_status = "NO_DATA"
    else:
        scopes = {"primary": PRIMARY_COHORTS, **{name: (name,) for name in DESCRIPTIVE_COHORTS}}
        for scope, names in scopes.items():
            blocks[scope] = width_contrast(
                [r for r in rows17 if r["cohort"] in names],
                [r for r in rows19 if r["cohort"] in names],
                gate=scope == "primary",
            )
        width_status = blocks["primary"]["contrasts"]["WIDTH_GAIN_24K"].get("gate", "UNDEFINED")
    result = {
        "statuses": {"LENGTH_STATUS": length_status, "WIDTH_STATUS": width_status},
        "INTERPRETATION_BRANCH": branch(length_status, width_status),
        "width_contrast": blocks,
        "v18": {**v18, "preregistration_sha256": frozen["preregistration_sha256"]},
        "input_sha256": {
            str(rows_path): sha(rows_path),
            str(root / "eval_results.json"): sha(root / "eval_results.json"),
        },
        "source_sha256": {
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        },
    }
    write_json(root / "width_contrast_results.json", result)
    print(
        "WIDTH_BRANCH", result["INTERPRETATION_BRANCH"], json.dumps(result["statuses"]), flush=True
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
