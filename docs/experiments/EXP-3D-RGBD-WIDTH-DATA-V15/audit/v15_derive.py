"""Derive the V15 pipeline (width x data) from the frozen V14 files (sources never modified).

Usage (repo root of a dry copy): python3 v15_derive.py. The EVAL stage script, the report and
the tests are written separately.
"""

from pathlib import Path


def read(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def write(target, text):
    path = Path(target)
    if path.exists():
        raise SystemExit(f"DERIVE FAILED {target}: target exists")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    print("derived", target)


def apply(text, target, replacements):
    for old, new, count in replacements:
        found = text.count(old)
        if found != count:
            raise SystemExit(f"DERIVE FAILED {target}: expected {count}, found {found}: {old[:80]!r}")
        text = text.replace(old, new)
    return text


def derive(source, target, replacements, leftovers=()):
    text = apply(read(source), target, replacements)
    for word in leftovers:
        if word in text:
            raise SystemExit(f"DERIVE FAILED {target}: leftover {word!r}")
    write(target, text)


LEFT = ("rgbd_data_scale", "data-scale", "DATA-SCALE", '"""V14', "V14 ", "DATA_GAIN", "train72_training")
CARRIER = (
    "from mcss.mechanism_pilot.rgbd_data_scale_carrier import ",
    "from mcss.mechanism_pilot.rgbd_wide_scale_carrier import ",
)
EVALUATION = (
    "from mcss.mechanism_pilot.rgbd_data_scale_evaluation import evaluate_model\n",
    "from mcss.mechanism_pilot.rgbd_wide_scale_evaluation import evaluate_model\n",
)
STREAM_COMMENT = (
    "        # V14 TRAIN sets differ by design; variants sharing a TRAIN set share the stream.\n",
    "        # V15 widths differ by design; both variants share TRAIN_EXT and so the stream.\n",
)
SRC = "src/mcss/mechanism_pilot/"

# ---------------------------------------------------------------- carrier module
derive(
    SRC + "rgbd_data_scale_carrier.py",
    SRC + "rgbd_wide_scale_carrier.py",
    [
        (
            '"""V14: more training scenes for completion, on the unchanged V11 C1 recipe (data scale).\n'
            "\n"
            "PLAN_BEFORE_TRAIN_EXT_DATA.md fixed V14 before any new training scene was read. C0 retrains\n"
            "the V11 C1 recipe on TRAIN72 (a machinery control that must equal V11 C1 bit-for-bit); C1 is\n"
            "the same recipe (width 32, 32^3, CONTEXT_DEPTH bounds, the frozen V2 C1 loss, 6000 steps, the\n"
            "V11 extra-view rule) trained on TRAIN72 plus the TRAIN-EXT scenes. Both variants share the\n"
            "V11 C1 parameter draw; only the variant's frozen TRAIN set differs. V8's train_records looks\n"
            "the TRAIN set up in the V8 variant table (TRAIN72 for every V8 variant), so the variant's own\n"
            "TRAIN set is resolved here instead.\n"
            '"""\n',
            '"""V15: carrier width 64 on the larger TRAIN72 + TRAIN-EXT set (width x data), V11 C1 recipe.\n'
            "\n"
            "PLAN_BEFORE_V15_TRAINING.md fixed V15 before any V15 training. Both variants train the V11 C1\n"
            "recipe (32^3, CONTEXT_DEPTH bounds, the frozen V2 C1 loss, 6000 steps, the V11 extra-view\n"
            "rule) on the 127 TRAIN72 + TRAIN-EXT scenes. C0 is width 32, i.e. V14 C1 (a machinery control\n"
            "that must equal it bit-for-bit); C1 is width 64 (hidden 64, expansion 128) with V12 C1's\n"
            "per-seed draw. Only the width differs; the variant's own TRAIN set is resolved from this\n"
            "table (V8's train_records reads the V8 table).\n"
            '"""\n',
            1,
        ),
        (
            'EXPERIMENT = "EXP-3D-RGBD-DATA-SCALE-V14"\nSCHEMA = "mcss.rgbd_data_scale_carrier.v14"\n',
            'EXPERIMENT = "EXP-3D-RGBD-WIDTH-DATA-V15"\nSCHEMA = "mcss.rgbd_wide_scale_carrier.v15"\n',
            1,
        ),
        (
            '    "C0": {"label": "TRAIN72", "hidden": 32, "expansion": 64, "model": "C1", "train": "TRAIN72"},\n'
            '    "C1": {\n'
            '        "label": "TRAIN_EXT",\n'
            '        "hidden": 32,\n'
            '        "expansion": 64,\n'
            '        "model": "C1",\n'
            '        "train": "TRAIN_EXT",\n'
            "    },\n",
            '    "C0": {\n'
            '        "label": "W32_T127",\n'
            '        "hidden": 32,\n'
            '        "expansion": 64,\n'
            '        "model": "C1",\n'
            '        "train": "TRAIN_EXT",\n'
            "    },\n"
            '    "C1": {\n'
            '        "label": "W64_T127",\n'
            '        "hidden": 64,\n'
            '        "expansion": 128,\n'
            '        "model": "C1",\n'
            '        "train": "TRAIN_EXT",\n'
            "    },\n",
            1,
        ),
        (
            'raise ValueError("The registered V14 width at 32^3 (one candidate per voxel) is required")',
            'raise ValueError("A registered V15 width at 32^3 (one candidate per voxel) is required")',
            1,
        ),
        ("frozen TRAIN set (V14 table)", "frozen TRAIN set (V15 table)", 1),
        (
            '    """The V8 resolution carrier at width 32; for both V14 variants this is the V11 C1 draw."""\n',
            '    """The V8 resolution carrier at the variant\'s width (32: V11 C1 draw, 64: V12 C1 draw)."""\n',
            1,
        ),
        (
            '    """The V8 build_state under the width guard of the registered V14 width."""\n',
            '    """The V8 build_state under the width guard of the two registered V15 widths."""\n',
            1,
        ),
        ('raise TypeError("V14 states are built from an RGBDContext")', 'raise TypeError("V15 states are built from an RGBDContext")', 1),
        (
            'raise PermissionError("V14 static RGB-D data-scale carrier schema required")',
            'raise PermissionError("V15 static RGB-D wide-scale carrier schema required")',
            1,
        ),
        (
            'raise PermissionError("V14 checkpoints always carry the measured-depth bypass")',
            'raise PermissionError("V15 checkpoints always carry the measured-depth bypass")',
            1,
        ),
    ],
    (
        "rgbd_data_scale",
        "data-scale",
        "DATA-SCALE",
        '"""V14',
        '"TRAIN72"',
        "V14 width",
        "V14 variants",
        "V14 states",
        "V14 static",
        "V14 checkpoints",
        "(V14 table)",
    ),
)

# ---------------------------------------------------------------- DEV evaluation module
derive(
    SRC + "rgbd_data_scale_evaluation.py",
    SRC + "rgbd_wide_scale_evaluation.py",
    [
        (
            '"""V14 DEV evaluation: the V11 Amendment 1 evaluator; only diagnostics runs save states.\n',
            '"""V15 DEV evaluation: the V11 Amendment 1 evaluator; only diagnostics runs save states.\n',
            1,
        ),
        (CARRIER[0] + "build_state, scene_data\n", CARRIER[1] + "build_state, scene_data\n", 1),
        ("    if diagnostics:  # V14: only", "    if diagnostics:  # V15: only", 1),
    ],
    (*LEFT, "V14"),
)

derive(
    "scripts/train_rgbd_data_scale_carrier.py",
    "scripts/train_rgbd_wide_scale_carrier.py",
    [
        (
            '"""V14 data-scale runs of the V11 C1 recipe; no writer or test cohort access."""',
            '"""V15 width-at-scale runs of the V11 C1 recipe; no writer or test cohort access."""',
            1,
        ),
        (CARRIER[0] + "(\n", CARRIER[1] + "(\n", 1),
        ("                " + EVALUATION[0], "                " + EVALUATION[1], 1),
    ],
    (*LEFT, "V14"),
)

derive(
    "scripts/evaluate_rgbd_data_scale_carrier.py",
    "scripts/evaluate_rgbd_wide_scale_carrier.py",
    [
        (
            '"""V14: evaluate frozen per-seed DEV selections and derive post-seal diagnostics."""',
            '"""V15: evaluate frozen per-seed DEV selections and derive post-seal diagnostics."""',
            1,
        ),
        (CARRIER[0] + "VARIANT_SPECS, load_checkpoint\n", CARRIER[1] + "VARIANT_SPECS, load_checkpoint\n", 1),
        (EVALUATION[0], EVALUATION[1], 1),
        (*STREAM_COMMENT, 1),
        (
            '        "C0": "train72_training",\n        "C1": "train_ext_training",\n',
            '        "C0": "width32_training",\n        "C1": "width64_training",\n',
            1,
        ),
    ],
    (*LEFT, "V14", "train_ext_training"),
)

derive(
    "scripts/diagnose_rgbd_data_scale_states.py",
    "scripts/diagnose_rgbd_wide_scale_states.py",
    [("on sealed V14 DEV states.", "on sealed V15 DEV states.", 1)],
    (*LEFT, "V14"),
)

derive(
    "scripts/analyze_rgbd_data_scale_carrier.py",
    "scripts/analyze_rgbd_wide_scale_carrier.py",
    [
        (
            '"""V14 raw-only DEV statistics (secondary) with the frozen V2 estimator and figures."""',
            '"""V15 raw-only DEV statistics (secondary) with the frozen V2 estimator and figures."""',
            1,
        ),
        (
            '        "C0": "C0 TRAIN72",\n        "C1": "C1 TRAIN72+EXT (primary)",\n',
            '        "C0": "C0 width 32",\n        "C1": "C1 width 64 (primary)",\n',
            1,
        ),
    ],
    (*LEFT, "V14", "TRAIN72+EXT"),
)

derive(
    "scripts/audit_rgbd_data_scale_statistics.py",
    "scripts/audit_rgbd_wide_scale_statistics.py",
    [
        ('"analyze_rgbd_data_scale_carrier.py"', '"analyze_rgbd_wide_scale_carrier.py"', 1),
        ('"rgbd_data_scale_reproduction_analyzer"', '"rgbd_wide_scale_reproduction_analyzer"', 1),
        ('prefix="rgbd-data-scale-statistics-audit-"', 'prefix="rgbd-wide-scale-statistics-audit-"', 1),
    ],
    LEFT,
)

derive(
    "scripts/reference_rgbd_data_scale_carrier.py",
    "scripts/reference_rgbd_wide_scale_carrier.py",
    [
        (
            '"""V14 preregistered geometry-free reference for the sealed selected DEV predictions."""',
            '"""V15 preregistered geometry-free reference for the sealed selected DEV predictions."""',
            1,
        ),
        (
            '    "V14": {\n'
            '        "labels": {"C0": "TRAIN72", "C1": "TRAIN_EXT"},\n'
            '        "primary": "DATA_GAIN_DEV",\n',
            '    "V15": {\n'
            '        "labels": {"C0": "W32_T127", "C1": "W64_T127"},\n'
            '        "primary": "WIDTH_GAIN_DEV",\n',
            1,
        ),
        ('                        "source": "V14",\n', '                        "source": "V15",\n', 1),
    ],
    (*LEFT, "V14", "TRAIN_EXT"),
)

derive(
    "scripts/run_rgbd_data_scale_experiment.py",
    "scripts/run_rgbd_wide_scale_experiment.py",
    [
        (
            '"""Execute the frozen V14 experiment (CPU only) and audits, never opening fresh data.',
            '"""Execute the frozen V15 experiment (CPU only) and audits, never opening fresh data.',
            1,
        ),
        ("rgbd_data_scale", "rgbd_wide_scale", 8),
        ('Path("/tmp/rgbd-data-scale-v14-finalization.log")', 'Path("/tmp/rgbd-wide-scale-v15-finalization.log")', 1),
    ],
    (*LEFT, "V14", "v14"),
)

derive(
    "scripts/finalize_rgbd_data_scale_carrier.py",
    "scripts/finalize_rgbd_wide_scale_carrier.py",
    [
        (
            '"""V14 independent postrun checks, preservation audit, and portable numeric archive."""',
            '"""V15 independent postrun checks, preservation audit, and portable numeric archive."""',
            1,
        ),
        (CARRIER[0] + "VARIANT_SPECS\n", CARRIER[1] + "VARIANT_SPECS\n", 1),
        ('if root.name != "EXP-3D-RGBD-DATA-SCALE-V14"', 'if root.name != "EXP-3D-RGBD-WIDTH-DATA-V15"', 1),
        (*STREAM_COMMENT, 1),
        (
            "    # any query depth, C0 equals the sealed V11 C1; results reproduce from rows.\n",
            "    # any query depth, C0 equals the sealed V14 C1; results reproduce from rows.\n",
            1,
        ),
        (
            '    if not seal_file["consistency"]["c0_minus_v11_c1_max_abs"] <= tolerance:\n'
            '        raise AssertionError("V14 C0 does not reproduce the sealed V11 C1 predictions")\n',
            '    if not seal_file["consistency"]["c0_minus_v14_c1_max_abs"] <= tolerance:\n'
            '        raise AssertionError("V15 C0 does not reproduce the sealed V14 C1 predictions")\n',
            1,
        ),
        (
            '        "v14_eval", Path("scripts/evaluate_rgbd_data_scale_eval.py")\n',
            '        "v15_eval", Path("scripts/evaluate_rgbd_wide_scale_eval.py")\n',
            1,
        ),
        ('            "scripts/report_rgbd_data_scale_carrier.py",\n', '            "scripts/report_rgbd_wide_scale_carrier.py",\n', 1),
        ('        Path("scripts").glob("*rgbd_data_scale*.py")\n', '        Path("scripts").glob("*rgbd_wide_scale*.py")\n', 1),
        (
            '    source += sorted(Path("tests").glob("test_rgbd_data_scale*.py"))\n',
            '    source += sorted(Path("tests").glob("test_rgbd_wide_scale*.py"))\n',
            1,
        ),
    ],
    ("rgbd_data_scale", "data-scale", "DATA-SCALE", '"""V14', "V14 C0", "v14_eval", "c0_minus_v11", "V11 C1;"),
)

exec(Path(__file__).with_name("v15_derive_prepare.py").read_text())
print("V15 DERIVATION COMPLETE")
