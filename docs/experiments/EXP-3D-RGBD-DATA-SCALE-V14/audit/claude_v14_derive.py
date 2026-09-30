"""Derive the V14 pipeline from the frozen V12 files (sources never modified).

Usage (repo root of a dry copy): python3 v14_derive.py. The EVAL stage script, the report and
the tests are written separately; the prepare, finalize and runner are derived here.
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


LEFT = ("rgbd_width", "WIDTH-V12", '"""V12', "width64", "WIDTH64", "width 64", "V12 ")
CARRIER = (
    "from mcss.mechanism_pilot.rgbd_width_carrier import ",
    "from mcss.mechanism_pilot.rgbd_data_scale_carrier import ",
)
EVALUATION = (
    "from mcss.mechanism_pilot.rgbd_width_evaluation import evaluate_model\n",
    "from mcss.mechanism_pilot.rgbd_data_scale_evaluation import evaluate_model\n",
)
STREAM_COMMENT = (
    "        # V12 widths differ by design, so only the data/ray stream must match per seed.\n",
    "        # V14 TRAIN sets differ by design; variants sharing a TRAIN set share the stream.\n",
)
TRAIN_SETS = (
    '            for train_set in ("TRAIN24", "TRAIN72")\n',
    '            for train_set in ("TRAIN24", "TRAIN72", "TRAIN_EXT")\n',
)

# ---------------------------------------------------------------- carrier module
derive(
    "src/mcss/mechanism_pilot/rgbd_width_carrier.py",
    "src/mcss/mechanism_pilot/rgbd_data_scale_carrier.py",
    [
        (
            '"""V12: carrier width 64 for completion, on the V11 C1 recipe fixed before any V11 result.\n'
            "\n"
            "PLAN_BEFORE_V11_RESULTS.md runs V12 only if the V11 EVAL-V3 COMPLETION_GAIN point estimate is\n"
            "> 0. C0 retrains the V11 C1 recipe at width 32 (a machinery control that must equal V11 C1\n"
            "bit-for-bit); C1 is the same recipe (TRAIN72, 32^3, CONTEXT_DEPTH bounds, the frozen V2 C1\n"
            "loss, 6000 steps, the V11 extra-view rule) with hidden_dim 64 and expansion_dim 128. The primary\n"
            "EVAL-V4 control is V11 C1's own selected checkpoints, loaded with the V11 loader. Every module\n"
            "width follows CarrierConfig, so the V8 computation, state construction and loss are reused;\n"
            "only the frozen width guard is generalized to the two V12 widths.\n"
            '"""\n',
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
            1,
        ),
        (
            'EXPERIMENT = "EXP-3D-RGBD-WIDTH-V12"\nSCHEMA = "mcss.rgbd_width_carrier.v12"\n',
            'EXPERIMENT = "EXP-3D-RGBD-DATA-SCALE-V14"\nSCHEMA = "mcss.rgbd_data_scale_carrier.v14"\n',
            1,
        ),
        (
            '    "C0": {"label": "WIDTH32", "hidden": 32, "expansion": 64, "model": "C1", "train": "TRAIN72"},\n'
            '    "C1": {"label": "WIDTH64", "hidden": 64, "expansion": 128, "model": "C1", "train": "TRAIN72"},\n',
            '    "C0": {"label": "TRAIN72", "hidden": 32, "expansion": 64, "model": "C1", "train": "TRAIN72"},\n'
            '    "C1": {\n'
            '        "label": "TRAIN_EXT",\n'
            '        "hidden": 32,\n'
            '        "expansion": 64,\n'
            '        "model": "C1",\n'
            '        "train": "TRAIN_EXT",\n'
            "    },\n",
            1,
        ),
        ("train_records = v8.train_records\n", "", 1),
        (
            "def carrier_config(variant):\n",
            "def train_records(manifest, config, variant):\n"
            '    """Exactly the variant\'s own frozen TRAIN set (V14 table), sorted by scene id."""\n'
            '    names = set(config["train_sets"][VARIANT_SPECS[variant]["train"]])\n'
            "    records = sorted(\n"
            '        (r for r in manifest["scenes"] if r["split"] == "TRAIN" and r["scene_id"] in names),\n'
            '        key=lambda r: r["scene_id"],\n'
            "    )\n"
            "    if len(records) != len(names) or len(records) < 24:\n"
            "        raise PermissionError(\"The variant's frozen TRAIN set is incomplete\")\n"
            "    return records\n"
            "\n"
            "\n"
            "def carrier_config(variant):\n",
            1,
        ),
        (
            '        raise ValueError("A registered V12 width at 32^3 (one candidate per voxel) is required")\n',
            '        raise ValueError("The registered V14 width at 32^3 (one candidate per voxel) is required")\n',
            1,
        ),
        (
            "    \"\"\"The V8 resolution carrier at the variant's width; at width 32 this is the V11 C1 draw.\"\"\"\n",
            '    """The V8 resolution carrier at width 32; for both V14 variants this is the V11 C1 draw."""\n',
            1,
        ),
        (
            '    """The V8 build_state with the width guard generalized to the registered V12 widths."""\n',
            '    """The V8 build_state under the width guard of the registered V14 width."""\n',
            1,
        ),
        (
            'raise TypeError("V12 states are built from an RGBDContext")',
            'raise TypeError("V14 states are built from an RGBDContext")',
            1,
        ),
        (
            'raise PermissionError("V12 static RGB-D width carrier schema required")',
            'raise PermissionError("V14 static RGB-D data-scale carrier schema required")',
            1,
        ),
        (
            'raise PermissionError("V12 checkpoints always carry the measured-depth bypass")',
            'raise PermissionError("V14 checkpoints always carry the measured-depth bypass")',
            1,
        ),
    ],
    (*LEFT, "V12", "v8.train_records"),
)

# ---------------------------------------------------------------- DEV evaluation module
derive(
    "src/mcss/mechanism_pilot/rgbd_width_evaluation.py",
    "src/mcss/mechanism_pilot/rgbd_data_scale_evaluation.py",
    [
        (
            '"""V12 DEV evaluation: the V11 Amendment 1 evaluator (float-rounding RGB clip), V12 widths.\n',
            '"""V14 DEV evaluation: the V11 Amendment 1 evaluator; only diagnostics runs save states.\n',
            1,
        ),
        (CARRIER[0] + "build_state, scene_data\n", CARRIER[1] + "build_state, scene_data\n", 1),
        (
            "    out.mkdir(parents=True)\n"
            '    torch.save({key: barrier.state(key) for key in keys}, out / "states.pt")\n',
            "    out.mkdir(parents=True)\n"
            "    if diagnostics:  # V14: only the selected-checkpoint evaluation keeps states (disk)\n"
            '        torch.save({key: barrier.state(key) for key in keys}, out / "states.pt")\n',
            1,
        ),
        (
            '            "states_file_sha256": sha(out / "states.pt"),\n',
            '            "states_file_sha256": sha(out / "states.pt") if diagnostics else None,\n',
            1,
        ),
    ],
    (*LEFT, "V12"),
)

# ---------------------------------------------------------------- training
derive(
    "scripts/train_rgbd_width_carrier.py",
    "scripts/train_rgbd_data_scale_carrier.py",
    [
        (
            '"""V12 carrier-width runs of the V11 C1 recipe; no writer or test cohort access."""',
            '"""V14 data-scale runs of the V11 C1 recipe; no writer or test cohort access."""',
            1,
        ),
        (CARRIER[0] + "(\n", CARRIER[1] + "(\n", 1),
        ("                " + EVALUATION[0], "                " + EVALUATION[1], 1),
    ],
    (*LEFT, "V12"),
)

# ---------------------------------------------------------------- selected DEV evaluation
derive(
    "scripts/evaluate_rgbd_width_carrier.py",
    "scripts/evaluate_rgbd_data_scale_carrier.py",
    [
        (
            '"""V12: evaluate frozen per-seed DEV selections and derive post-seal diagnostics."""',
            '"""V14: evaluate frozen per-seed DEV selections and derive post-seal diagnostics."""',
            1,
        ),
        (CARRIER[0] + "VARIANT_SPECS, load_checkpoint\n", CARRIER[1] + "VARIANT_SPECS, load_checkpoint\n", 1),
        (EVALUATION[0], EVALUATION[1], 1),
        (*STREAM_COMMENT, 1),
        (*TRAIN_SETS, 1),
        (
            '        "C0": "width32_training",\n        "C1": "width64_training",\n',
            '        "C0": "train72_training",\n        "C1": "train_ext_training",\n',
            1,
        ),
    ],
    (*LEFT, "V12"),
)

derive(
    "scripts/diagnose_rgbd_width_states.py",
    "scripts/diagnose_rgbd_data_scale_states.py",
    [
        (
            '"""Preregistered descriptive state-correlation diagnostic on sealed V12 DEV states.',
            '"""Preregistered descriptive state-correlation diagnostic on sealed V14 DEV states.',
            1,
        )
    ],
    (*LEFT, "V12"),
)

derive(
    "scripts/analyze_rgbd_width_carrier.py",
    "scripts/analyze_rgbd_data_scale_carrier.py",
    [
        (
            '"""V12 raw-only DEV statistics (secondary) with the frozen V2 estimator and figures."""',
            '"""V14 raw-only DEV statistics (secondary) with the frozen V2 estimator and figures."""',
            1,
        ),
        (
            '        "C0": "C0 width 32",\n        "C1": "C1 width 64 (primary)",\n',
            '        "C0": "C0 TRAIN72",\n        "C1": "C1 TRAIN72+EXT (primary)",\n',
            1,
        ),
    ],
    (*LEFT, "V12"),
)

derive(
    "scripts/audit_rgbd_width_statistics.py",
    "scripts/audit_rgbd_data_scale_statistics.py",
    [
        ('"analyze_rgbd_width_carrier.py"', '"analyze_rgbd_data_scale_carrier.py"', 1),
        ('"rgbd_width_reproduction_analyzer"', '"rgbd_data_scale_reproduction_analyzer"', 1),
        ('prefix="rgbd-width-statistics-audit-"', 'prefix="rgbd-data-scale-statistics-audit-"', 1),
    ],
    (*LEFT, "V12", "rgbd-width"),
)

derive(
    "scripts/reference_rgbd_width_carrier.py",
    "scripts/reference_rgbd_data_scale_carrier.py",
    [
        (
            '"""V12 preregistered geometry-free reference for the sealed selected DEV predictions."""',
            '"""V14 preregistered geometry-free reference for the sealed selected DEV predictions."""',
            1,
        ),
        (
            '    "V12": {\n'
            '        "labels": {"C0": "WIDTH32", "C1": "WIDTH64"},\n'
            '        "primary": "WIDTH_GAIN_DEV",\n',
            '    "V14": {\n'
            '        "labels": {"C0": "TRAIN72", "C1": "TRAIN_EXT"},\n'
            '        "primary": "DATA_GAIN_DEV",\n',
            1,
        ),
        ('                        "source": "V12",\n', '                        "source": "V14",\n', 1),
    ],
    (*LEFT, "V12", "WIDTH"),
)

exec(Path(__file__).with_name("v14_derive_part2.py").read_text())
