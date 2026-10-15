"""EXP-3D-CAJU3D-ENGINE-V1 (CAJU-3D step 1): frozen constants, cohort selection and the decision rules.

Free of torch and DepthSplat. The engine variants themselves are defined in engine_core.py (math) and engine_ttt.py
(the GPU replay); the readout is CAJU Gate 0's caju_logic.core_b (V3's Core B readout).
"""

import random

SCALE_MAX, STEPS, MULT, SEED, ALPHA_MIN = 0.1, 100, 1.0, 20261008, 0.5
RELEASED = ("/home/xmum26_zonghan/zonghan_o_mirror/_models/depthsplat/"
            "depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth")
JN1 = ("/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state/outputs/"
       "EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1/checkpoints/JN1/depthsplat_jepa_final.pth")
CHECKPOINTS = {"ZS": RELEASED, "JN1": JN1}
VARIANTS = ("RAW", "NORM", "NORM_DEC", "NORM_DEC_INC")
DEV_ARMS = {"ZS": ("CACHE", "NORM", "NORM_DEC", "NORM_DEC_INC"),       # ZS CACHE / RAW are also sealed (V1, V2)
            "JN1": ("CACHE", "RAW", "NORM", "NORM_DEC", "NORM_DEC_INC")}  # JN1 CACHE is also sealed (CA)
CANDIDATES = ("ZS_NORM", "ZS_NORM_DEC", "ZS_NORM_DEC_INC", "JN1_RAW", "JN1_NORM", "JN1_NORM_DEC", "JN1_NORM_DEC_INC")
COMPARATOR = "ZS_RAW"  # Core B: released weights, V1's TTT, Core B readout
INC = {"cos_lo": 0.1, "cos_hi": 0.4, "edge_max": 0.1, "anchor_conf_opacity": 0.65}
DEV_SEED, DEV_PER_COHORT = 20261015, 12
DEV_COHORTS = {"DEV_SYSTEM": ("EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1", "ARKIT_SYSTEM"),
               "DEV_SYSTEM2": ("EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2", "ARKIT_SYSTEM2")}
CONF_COHORTS = {"CONF_SYSTEM3": ("EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3", "ARKIT_SYSTEM3"),
                "CONF_VAL": ("EXP-3D-RGBD-ENSEMBLES-ARKIT-V1", "ARKIT_VAL")}


def select_dev_scans(scene_ids, cohort_index):
    """12 scans of one source cohort, seeded per cohort (sorted ids, random.Random(DEV_SEED + index))."""
    ids = sorted(set(scene_ids))
    random.Random(DEV_SEED + cohort_index).shuffle(ids)
    return sorted(ids[:DEV_PER_COHORT])


def select_candidate(dev_means):
    """The candidate with the lowest DEV mean place AbsRel (Core B readout); ties to the earlier one in CANDIDATES.
    STOP when it does not beat the comparator (Core B) on DEV."""
    best = min(CANDIDATES, key=lambda k: (round(dev_means[k], 12), CANDIDATES.index(k)))
    stop = not dev_means[best] < dev_means[COMPARATOR]
    return best, stop


def arm_of(candidate):
    """('JN1', 'NORM') from 'JN1_NORM'."""
    model, _, variant = candidate.partition("_")
    if model not in CHECKPOINTS or variant not in VARIANTS:
        raise ValueError(candidate)
    return model, variant


def branch(primary_status, replication):
    """Final branch from the V2 status of P1 (SYSTEM3) and the replication P2 (VAL)."""
    if primary_status == "SUPPORTED" and replication["mean"] > 0:
        return "CAJU3D_ENGINE_SUPPORTED"
    if primary_status in ("SUPPORTED", "PARTIAL"):
        return "CAJU3D_ENGINE_PARTIAL"
    if primary_status == "HARMFUL":
        return "CAJU3D_ENGINE_HARMFUL"
    return "CAJU3D_ENGINE_NOT_ESTABLISHED"
