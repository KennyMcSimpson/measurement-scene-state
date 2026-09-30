# Part 2 of v15_derive.py (executed in its namespace): the freezing prepare script.

derive(
    "scripts/prepare_rgbd_data_scale_experiment.py",
    "scripts/prepare_rgbd_wide_scale_experiment.py",
    [
        (
            '"""Freeze the V14 data-scale protocol (TRAIN72 vs TRAIN72+EXT, EVAL-V3/V4 primary, CPU)."""',
            '"""Freeze the V15 width-at-scale protocol (width 32 vs 64 on TRAIN127, EVAL-V3/V4 primary)."""',
            1,
        ),
        (CARRIER[0] + "(\n", CARRIER[1] + "(\n", 1),
        (
            'V12 = Path("outputs/EXP-3D-RGBD-WIDTH-V12")\n'
            'TRAIN_EXT = Path("outputs/EXP-3D-RGBD-TRAIN-EXT-DATA")\n',
            'V12 = Path("outputs/EXP-3D-RGBD-WIDTH-V12")\n'
            'V14 = Path("outputs/EXP-3D-RGBD-DATA-SCALE-V14")\n'
            'TRAIN_EXT = Path("outputs/EXP-3D-RGBD-TRAIN-EXT-DATA")\n',
            1,
        ),
        (
            "# Sealed V11 C1 predictions that V14 C0 must reproduce (V11, the EVAL-V4 replication, V13).\n"
            "V11_C1_SEALS = {\n"
            '    "EVAL_V3": V11 / "raw/eval_v3/prediction_seal.json",\n'
            '    "EVAL_V4": Path("outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/raw/prediction_seal.json"),\n'
            '    "EVAL_FAR": Path("outputs/EXP-3D-RGBD-VOLUME-V13/raw/EVAL_FAR/prediction_seal.json"),\n'
            "}\n"
            'PLAN = "PLAN_BEFORE_TRAIN_EXT_DATA.md"\n'
            'PLAN_SHA256 = "526c3623752317da05922d5368e341938ba9fc7b660b7e504746e910ae6eb1b6"\n'
            "# Experiments that may legitimately write while V14 runs; never sealed, listed in the seal.\n",
            "# The sealed V14 predictions of every cohort: V15 C0 must reproduce V14 C1 (width 32,\n"
            "# TRAIN127); V14 C0 (width 32, TRAIN72 = V11 C1) is the descriptive 2x2 cell.\n"
            'V14_SEAL = V14 / "raw/eval/prediction_seal.json"\n'
            'V14_PLAN_SHA256 = "526c3623752317da05922d5368e341938ba9fc7b660b7e504746e910ae6eb1b6"\n'
            'PLAN = "PLAN_BEFORE_V15_TRAINING.md"\n'
            'PLAN_SHA256 = "4c0cc1b719b6dbc5459075ac36a89b637536b05d08b80867a5b485f31fb7beb6"\n'
            "# Experiments that may legitimately write while V15 runs; never sealed, listed in the seal.\n",
            1,
        ),
        ('COUNTS = {"C0": 64238, "C1": 64238}\n', 'COUNTS = {"C0": 64238, "C1": 250542}\n', 1),
        ('raise PermissionError("Exact V14 experiment directory required")', 'raise PermissionError("Exact V15 experiment directory required")', 1),
        (
            'raise PermissionError("Frozen V14 protocol text required before freezing")',
            'raise PermissionError("Frozen V15 protocol text required before freezing")',
            1,
        ),
        (
            "    for earlier in (V2, V3, V4, V5, V6, V7, V8, V9, V11, V12):\n"
            '        if read(earlier / "integrity.json")["status"] != "PASS":\n'
            '            raise PermissionError(f"{earlier.name} must be finalized before V14 freezes")\n'
            "    if sha(root / PLAN) != PLAN_SHA256:\n"
            '        raise PermissionError("The V14 plan written before the TRAIN-EXT data must be intact")\n',
            "    for earlier in (V2, V3, V4, V5, V6, V7, V8, V9, V11, V12, V14):\n"
            '        if read(earlier / "integrity.json")["status"] != "PASS":\n'
            '            raise PermissionError(f"{earlier.name} must be finalized before V15 freezes")\n'
            "    if sha(root / PLAN) != PLAN_SHA256:\n"
            '        raise PermissionError("The V15 plan written before any V15 training must be intact")\n',
            1,
        ),
        ('        ext_lock["plan_sha256"] != PLAN_SHA256\n', '        ext_lock["plan_sha256"] != V14_PLAN_SHA256\n', 1),
        (
            '        raise PermissionError("The V14 plan trains only with >= 40 valid TRAIN-EXT scenes")\n',
            '        raise PermissionError("V15 trains on the >= 40 valid TRAIN-EXT scenes of V14")\n',
            1,
        ),
        ('raise PermissionError("V14 keeps the V11 grid")', 'raise PermissionError("V15 keeps the V11 grid")', 1),
        (
            '            raise PermissionError(f"V11 C1 selected checkpoint changed: {seed}")\n',
            '            raise PermissionError(f"V11 C1 selected checkpoint changed: {seed}")\n'
            '    v12_selected = V12 / "selected_checkpoints.json"\n'
            '    for seed, item in read(v12_selected)["C1"].items():\n'
            '        if sha(Path(item["path"])) != item["sha256"]:\n'
            '            raise PermissionError(f"V12 C1 selected checkpoint changed: {seed}")\n'
            "    v14_seal = read(V14_SEAL)\n"
            '    if v14_seal["query_depth_read"] or v14_seal["consistency"]["c0_minus_v11_c1_max_abs"]:\n'
            '        raise PermissionError("The sealed V14 predictions (C0 = V11 C1) are required")\n',
            1,
        ),
        (
            '        sealed = read(V11_C1_SEALS[name])["predictions"]\n'
            '        if len(sealed) != 4 * len(scenes) or read(V11_C1_SEALS[name])["query_depth_read"]:\n'
            '            raise PermissionError(f"The sealed V11 C1 {name} predictions are required")\n',
            '        sealed = [k for k in v14_seal["predictions"] if k.startswith(f"{name}/")]\n'
            "        if len(sealed) != 4 * len(scenes):\n"
            '            raise PermissionError(f"The sealed V14 {name} predictions are required")\n',
            1,
        ),
        (
            '            "v11_c1_seal": str(V11_C1_SEALS[name].resolve()),\n',
            '            "v14_seal": str(V14_SEAL.resolve()),\n',
            1,
        ),
        (
            '    v11_hashes = read(V11 / "architecture_contract.json")["parameter_hash_by_seed"]\n'
            '    if {str(s): h["C0"] for s, h in hashes.items()} != {s: h["C1"] for s, h in v11_hashes.items()}:\n'
            '        raise AssertionError("V14 C0 must reuse the V11 C1 per-seed parameter draw")\n'
            '    if any(h["C0"] != h["C1"] for h in hashes.values()):\n'
            '        raise AssertionError("Both V14 variants must share the V11 C1 per-seed draw")\n',
            '    v11_hashes = read(V11 / "architecture_contract.json")["parameter_hash_by_seed"]\n'
            '    v12_hashes = read(V12 / "architecture_contract.json")["parameter_hash_by_seed"]\n'
            '    if {str(s): h["C0"] for s, h in hashes.items()} != {s: h["C1"] for s, h in v11_hashes.items()}:\n'
            '        raise AssertionError("V15 C0 must reuse the V11 C1 per-seed parameter draw")\n'
            '    if {str(s): h["C1"] for s, h in hashes.items()} != {s: h["C1"] for s, h in v12_hashes.items()}:\n'
            '        raise AssertionError("V15 C1 must reuse the V12 C1 (width 64) per-seed draw")\n',
            1,
        ),
        (
            '            "only_difference_C0_C1": "C1 differs from C0 only by its frozen TRAIN set: "\n'
            '            f"TRAIN72 plus {len(ext_records)} TRAIN-EXT scenes instead of TRAIN72 (identical "\n'
            '            "architecture, 64238 parameters and per-seed draw). "\n'
            '            f"The recipe is the {base_variant} one: width 32, "\n',
            '            "only_difference_C0_C1": "C1 differs from C0 only by the carrier width: hidden_dim "\n'
            '            "64 and expansion_dim 128 instead of 32 and 64 (250542 instead of 64238 "\n'
            '            "parameters; the per-seed draws are V12 C1\'s and V11 C1\'s). Both train on TRAIN72 "\n'
            '            f"plus {len(ext_records)} TRAIN-EXT scenes. "\n'
            '            f"The recipe is the {base_variant} one: "\n',
            1,
        ),
        (
            '            f"+ {EXTRA_SEED_OFFSET}); the scene permutation covers the variant\'s own TRAIN "\n'
            '            "set, so C1\'s scene order, query slots and ray indices differ from C0\'s. "\n'
            '            "C0 retrains V11 C1 and must reproduce it bit-for-bit. DEV evaluation and "\n',
            '            f"+ {EXTRA_SEED_OFFSET}); scene order, query slots and ray indices are identical. "\n'
            '            "C0 retrains V14 C1 and must reproduce it bit-for-bit. DEV evaluation and "\n',
            1,
        ),
        (
            '        "primary_contrast": "pooled EVAL-V3 + EVAL-V4 DATA_GAIN = FAR AbsRel(C0 TRAIN72) - "\n'
            '        "FAR AbsRel(C1 TRAIN72+EXT) (frozen V2 gate) and HARMONIC_HYBRID_GAIN = "\n'
            '        "AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 of C1) (INPAINT-V1 label rule)",\n'
            '        "secondary_contrast": "DEV DATA_GAIN_DEV = AbsRel(C0) - AbsRel(C1), frozen V2 "\n',
            '        "primary_contrast": "pooled EVAL-V3 + EVAL-V4 WIDTH_AT_SCALE_GAIN = FAR AbsRel(C0 "\n'
            '        "width 32) - FAR AbsRel(C1 width 64), both on TRAIN72+EXT (frozen V2 gate) and "\n'
            '        "HARMONIC_HYBRID_GAIN = AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 of C1) (INPAINT-V1 "\n'
            '        "label rule)",\n'
            '        "secondary_contrast": "DEV WIDTH_GAIN_DEV = AbsRel(C0) - AbsRel(C1), frozen V2 "\n',
            1,
        ),
        (
            '        "base_basis": f"{PLAN} (sha256 {PLAN_SHA256}), written before any TRAIN-EXT scene "\n'
            '        "was read: the unchanged V11 C1 recipe, only the TRAIN set grows",\n',
            '        "base_basis": f"{PLAN} (sha256 {PLAN_SHA256}), written before any V15 training: "\n'
            '        "the V14 C1 recipe (V11 C1 on TRAIN72+EXT) at widths 32 and 64",\n',
            1,
        ),
        (
            '            "c0_reproduction_tolerance": 1e-6,\n',
            '            "c0_reproduction_tolerance": 1e-6,\n'
            '            "width64_train72": {\n'
            '                "selected": str(v12_selected.resolve()),\n'
            '                "selected_sha256": sha(v12_selected),\n'
            '                "method": "V12 C1 selected checkpoints (width 64, TRAIN72), V12 loader; "\n'
            '                "the descriptive 2x2 cell, rendered on the primary cohorts only",\n'
            "            },\n"
            '            "last_step": 6000,\n',
            1,
        ),
        (
            '        "initialize": "from scratch; torch.manual_seed(seed) then the width-32 carrier; C0 "\n'
            '        "and C1 equal the V11 C1 draw; the depth bypass starts at zero without consuming "\n'
            '        "the RNG",\n',
            '        "initialize": "from scratch; torch.manual_seed(seed) then the width\'s carrier; C0 "\n'
            '        "equals the V11 C1 draw and C1 the V12 C1 draw; the depth bypass starts at zero "\n'
            '        "without consuming the RNG",\n',
            1,
        ),
        (
            '        "sampling": "seeded permutation of the variant\'s own TRAIN set, cycled; primary query "\n'
            '        "slot cycles per scene visit; uniform with-replacement rays, identical indices for "\n'
            '        "A/B; the extra-view count comes from its own generator so the ray stream is "\n'
            '        "unchanged",\n',
            '        "sampling": "seeded permutation of the shared TRAIN72+EXT set, cycled; primary query "\n'
            '        "slot cycles per scene visit; uniform with-replacement rays, identical indices for "\n'
            '        "A/B and both widths; the extra-view count comes from its own generator so the ray "\n'
            '        "stream is unchanged",\n',
            1,
        ),
        (
            '        "execution": "all six runs (2 TRAIN sets x 3 seeds) concurrently, one core each, on "\n',
            '        "execution": "all six runs (2 widths x 3 seeds) concurrently, one core each, on "\n',
            1,
        ),
        (
            '        "denote the secondary DEV C0-C1 contrast, i.e. DATA_GAIN_DEV in V14",\n'
            '        "dev_reuse": "the 8 DEV scenes were evaluated in V2-V12 and Core B V1/V2; in V14 "\n',
            '        "denote the secondary DEV C0-C1 contrast, i.e. WIDTH_GAIN_DEV in V15",\n'
            '        "dev_reuse": "the 8 DEV scenes were evaluated in V2-V14 and Core B V1/V2; in V15 "\n',
            1,
        ),
        (
            '            "authority": "mcss.mechanism_pilot.rgbd_data_scale_carrier.training_loss: the V8 "\n',
            '            "authority": "mcss.mechanism_pilot.rgbd_wide_scale_carrier.training_loss: the V8 "\n',
            1,
        ),
        ('            "DATA_DEV_STATUS": "secondary, DEV:', '            "WIDTH_DEV_STATUS": "secondary, DEV:', 1),
        (
            '                "DATA_STATUS": "frozen V2 gate checks (mean>0, CIlo>0, >=.75 "\n'
            '                "nonworse, all LOSO>0, positive top1 share<=.5) on DATA_GAIN = "\n',
            '                "WIDTH_AT_SCALE_STATUS": "frozen V2 gate checks (mean>0, CIlo>0, >=.75 "\n'
            '                "nonworse, all LOSO>0, positive top1 share<=.5) on WIDTH_AT_SCALE_GAIN = "\n',
            1,
        ),
        (
            '                "HFILL8_DATA_STATUS": "frozen V2 gate checks on AbsRel(HFILL8 of C0) - "\n',
            '                "HFILL8_WIDTH_STATUS": "frozen V2 gate checks on AbsRel(HFILL8 of C0) - "\n',
            1,
        ),
        (
            '                "C0_REPRODUCES_V11_C1": "C0 predictions equal the sealed V11 C1 predictions "\n'
            '                "of every cohort (max abs <= 1e-6) before any query depth is read, else stop",\n',
            '                "C0_REPRODUCES_V14_C1": "C0 predictions equal the sealed V14 C1 predictions "\n'
            '                "of every cohort (max abs <= 1e-6) before any query depth is read, else stop",\n'
            '                "TWO_BY_TWO": "descriptive: width 32 / TRAIN72 (the sealed V14 C0 "\n'
            '                "predictions) and width 64 / TRAIN72 (V12 C1 selected checkpoints) on the same "\n'
            '                "scenes; interaction [FAR(32,72) - FAR(64,72)] - [FAR(32,127) - FAR(64,127)]",\n'
            '                "LAST_STEP": "descriptive: the step-6000 checkpoints of C0 and C1",\n',
            1,
        ),
        (
            '                "A": "DATA_STATUS SUPPORTED and HARMONIC_HYBRID_LABEL ABOVE on the pooled "\n'
            '                "EVAL-V3 + EVAL-V4: more training scenes improve completion, and the carrier "\n'
            '                "trained on them beats classical inpainting inside the deployable hybrid",\n',
            '                "A": "WIDTH_AT_SCALE_STATUS SUPPORTED and HARMONIC_HYBRID_LABEL ABOVE on the "\n'
            '                "pooled EVAL-V3 + EVAL-V4: with more training scenes, width improves "\n'
            '                "completion and the wider carrier beats classical inpainting in the hybrid",\n',
            1,
        ),
        (
            '    source_paths += sorted(Path("scripts").glob("*rgbd_data_scale*.py"))\n',
            '    source_paths += sorted(Path("scripts").glob("*rgbd_wide_scale*.py"))\n',
            1,
        ),
    ],
    (
        "rgbd_data_scale",
        "data-scale",
        '"""V14',
        "DATA_GAIN",
        "DATA_STATUS",
        "DATA_DEV",
        "V11_C1_SEALS",
        "v11_c1_seal",
        "before V14 freezes",
        "V14 plan trains",
        "in V14",
    ),
)
