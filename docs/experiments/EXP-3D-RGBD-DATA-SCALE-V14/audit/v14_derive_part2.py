# Part 2 of v14_derive.py (executed in its namespace): prepare, finalize and runner.

# ---------------------------------------------------------------- prepare (freeze)
derive(
    "scripts/prepare_rgbd_width_experiment.py",
    "scripts/prepare_rgbd_data_scale_experiment.py",
    [
        (
            '"""Freeze the V12 carrier-width protocol (TRAIN72 training, EVAL-V4 primary, CPU only)."""',
            '"""Freeze the V14 data-scale protocol (TRAIN72 vs TRAIN72+EXT, EVAL-V3/V4 primary, CPU)."""',
            1,
        ),
        (CARRIER[0] + "(\n", CARRIER[1] + "(\n", 1),
        (
            'V11 = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")\n'
            'EVAL_V4 = Path("outputs/EXP-3D-RGBD-EVAL-V4-DATA")\n'
            'PLAN = "PLAN_BEFORE_V11_RESULTS.md"\n'
            'PLAN_SHA256 = "b726d336a6b6377f49d75c991e228a21380092312d6ee5e9b6f57828f639fb1c"\n'
            "# Experiments that may legitimately write while V12 runs; never sealed, listed in the seal.\n"
            'CONCURRENT = ("EXP-3D-RGB-PRIOR-BOUNDS-V10", "EXP-3D-RGBD-REPLICATION-EVAL-V4")\n',
            'V11 = Path("outputs/EXP-3D-RGBD-COMPLETION-V11")\n'
            'V12 = Path("outputs/EXP-3D-RGBD-WIDTH-V12")\n'
            'TRAIN_EXT = Path("outputs/EXP-3D-RGBD-TRAIN-EXT-DATA")\n'
            "TRAIN_EXT_MIN = 40\n"
            "# Evaluation cohorts: data directory, manifest, query role; EVAL-V3/V4 are pooled (primary).\n"
            "EVAL = {\n"
            '    "EVAL_V3": (Path("outputs/EXP-3D-RGBD-EVAL-V3-DATA"), "manifest_eval_v3.json", "primary_query"),\n'
            '    "EVAL_V4": (Path("outputs/EXP-3D-RGBD-EVAL-V4-DATA"), "manifest_eval_v4.json", "primary_query"),\n'
            '    "EVAL_FAR": (Path("outputs/EXP-3D-RGBD-EVAL-FAR-DATA"), "manifest_eval_far.json", "far_query"),\n'
            "}\n"
            "# Sealed V11 C1 predictions that V14 C0 must reproduce (V11, the EVAL-V4 replication, V13).\n"
            "V11_C1_SEALS = {\n"
            '    "EVAL_V3": V11 / "raw/eval_v3/prediction_seal.json",\n'
            '    "EVAL_V4": Path("outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/raw/prediction_seal.json"),\n'
            '    "EVAL_FAR": Path("outputs/EXP-3D-RGBD-VOLUME-V13/raw/EVAL_FAR/prediction_seal.json"),\n'
            "}\n"
            'PLAN = "PLAN_BEFORE_TRAIN_EXT_DATA.md"\n'
            'PLAN_SHA256 = "526c3623752317da05922d5368e341938ba9fc7b660b7e504746e910ae6eb1b6"\n'
            "# Experiments that may legitimately write while V14 runs; never sealed, listed in the seal.\n"
            "CONCURRENT = ()\n",
            1,
        ),
        ('COUNTS = {"C0": 64238, "C1": 250542}\n', 'COUNTS = {"C0": 64238, "C1": 64238}\n', 1),
        (
            'raise PermissionError("Exact V12 experiment directory required")',
            'raise PermissionError("Exact V14 experiment directory required")',
            1,
        ),
        (
            'raise PermissionError("Frozen V12 protocol text required before freezing")',
            'raise PermissionError("Frozen V14 protocol text required before freezing")',
            1,
        ),
        (
            "    for earlier in (V2, V3, V4, V5, V6, V7, V8, V9, V11):\n"
            '        if read(earlier / "integrity.json")["status"] != "PASS":\n'
            '            raise PermissionError(f"{earlier.name} must be finalized before V12 freezes")\n'
            "    if sha(root / PLAN) != PLAN_SHA256:\n"
            '        raise PermissionError("The pre-V11 plan deciding V12 must be intact")\n'
            '    gain = float(read(V11 / "eval_v3_results.json")["COMPLETION_GAIN"]["mean"])\n'
            "    if not gain > 0:\n"
            '        raise PermissionError("The pre-V11 plan runs V12 only if COMPLETION_GAIN > 0")\n',
            "    for earlier in (V2, V3, V4, V5, V6, V7, V8, V9, V11, V12):\n"
            '        if read(earlier / "integrity.json")["status"] != "PASS":\n'
            '            raise PermissionError(f"{earlier.name} must be finalized before V14 freezes")\n'
            "    if sha(root / PLAN) != PLAN_SHA256:\n"
            '        raise PermissionError("The V14 plan written before the TRAIN-EXT data must be intact")\n'
            '    ext_lock = read(TRAIN_EXT / "candidate_lock.json")\n'
            '    ext_integrity = read(TRAIN_EXT / "preparation_integrity.json")\n'
            "    if (\n"
            '        ext_lock["plan_sha256"] != PLAN_SHA256\n'
            '        or ext_integrity["status"] != "PASS"\n'
            '        or ext_integrity["candidate_lock_sha256"] != sha(TRAIN_EXT / "candidate_lock.json")\n'
            "    ):\n"
            '        raise PermissionError("TRAIN-EXT must PASS against its frozen lock under the V14 plan")\n'
            '    ext_records = read(TRAIN_EXT / "manifest_train_ext.json")["scenes"]\n'
            '    if len(ext_records) < TRAIN_EXT_MIN or any(r["split"] != "TRAIN" for r in ext_records):\n'
            '        raise PermissionError("The V14 plan trains only with >= 40 valid TRAIN-EXT scenes")\n',
            1,
        ),
        ('raise PermissionError("V12 keeps the V11 grid")', 'raise PermissionError("V14 keeps the V11 grid")', 1),
        (
            '    eval_integrity = read(EVAL_V4 / "preparation_integrity.json")\n'
            '    eval_manifest = EVAL_V4 / "manifest_eval_v4.json"\n'
            '    if eval_integrity["status"] != "PASS" or eval_integrity["candidate_lock_sha256"] != sha(\n'
            '        EVAL_V4 / "candidate_lock.json"\n'
            "    ):\n"
            '        raise PermissionError("EVAL-V4 preparation must PASS against its frozen lock")\n'
            '    eval_scenes = read(eval_manifest)["scenes"]\n'
            '    if len(eval_scenes) < 20 or any(r["split"] != "EVAL_V4" for r in eval_scenes):\n'
            '        raise PermissionError("EVAL-V4 needs at least 20 verified EVAL_V4 scenes")\n',
            "    cohorts = {}\n"
            "    for name, (directory, filename, role) in EVAL.items():\n"
            '        integrity = read(directory / "preparation_integrity.json")\n'
            '        if integrity["status"] != "PASS" or integrity["candidate_lock_sha256"] != sha(\n'
            '            directory / "candidate_lock.json"\n'
            "        ):\n"
            '            raise PermissionError(f"{name} preparation must PASS against its frozen lock")\n'
            '        scenes = read(directory / filename)["scenes"]\n'
            '        if len(scenes) < 20 or any(r["split"] != name for r in scenes):\n'
            '            raise PermissionError(f"{name} needs at least 20 verified {name} scenes")\n'
            '        sealed = read(V11_C1_SEALS[name])["predictions"]\n'
            '        if len(sealed) != 4 * len(scenes) or read(V11_C1_SEALS[name])["query_depth_read"]:\n'
            '            raise PermissionError(f"The sealed V11 C1 {name} predictions are required")\n'
            "        cohorts[name] = {\n"
            '            "manifest": str((directory / filename).resolve()),\n'
            '            "manifest_sha256": sha(directory / filename),\n'
            '            "scenes": len(scenes),\n'
            '            "role": role,\n'
            '            "v11_c1_seal": str(V11_C1_SEALS[name].resolve()),\n'
            "        }\n",
            1,
        ),
        (
            '    train72 = sorted(r["scene_id"] for r in manifest["scenes"] if r["split"] == "TRAIN")\n'
            "    validate_split(manifest)\n"
            '    counts = {s: sum(r["split"] == s for r in manifest["scenes"]) for s in ("TRAIN", "DEV")}\n'
            '    if counts != {"TRAIN": 72, "DEV": 8} or len(train24) != 24:\n'
            '        raise PermissionError(f"Unexpected V2 cohort: {counts}")\n'
            '    write_json(root / "scene_split.json", manifest)\n',
            '    train72 = sorted(r["scene_id"] for r in manifest["scenes"] if r["split"] == "TRAIN")\n'
            '    counts = {s: sum(r["split"] == s for r in manifest["scenes"]) for s in ("TRAIN", "DEV")}\n'
            '    if counts != {"TRAIN": 72, "DEV": 8} or len(train24) != 24:\n'
            '        raise PermissionError(f"Unexpected V2 cohort: {counts}")\n'
            '    manifest["scenes"] = manifest["scenes"] + ext_records\n'
            '    train_ext = sorted(r["scene_id"] for r in manifest["scenes"] if r["split"] == "TRAIN")\n'
            "    validate_split(manifest)\n"
            '    dev_volumes = {r["scene_id"].split("_")[1] for r in manifest["scenes"] if r["split"] == "DEV"}\n'
            "    if len(train_ext) != 72 + len(ext_records) or any(\n"
            '        r["scene_id"].split("_")[1] in dev_volumes for r in ext_records\n'
            "    ):\n"
            '        raise PermissionError("TRAIN-EXT must add new scenes outside every DEV volume")\n'
            '    physical = {r["physical_scene_id"].casefold() for r in manifest["scenes"]}\n'
            "    for name, entry in cohorts.items():\n"
            '        for r in read(entry["manifest"])["scenes"]:\n'
            '            if r["scene_id"] in train_ext or r["physical_scene_id"].casefold() in physical:\n'
            '                raise PermissionError(f"{name} shares a scene or source asset with training")\n'
            '    write_json(root / "scene_split.json", manifest)\n',
            1,
        ),
        (
            '            "source": "V2 locked cohort, frame roles and GT-free bounds rule reused unchanged",\n',
            '            "source": "V2 locked cohort plus the TRAIN-EXT records (frozen camera-only role "\n'
            '            "rule); the GT-free bounds rule is reused unchanged",\n',
            1,
        ),
        (
            '        raise AssertionError("V12 C0 must reuse the V11 C1 per-seed parameter draw")\n'
            '    if any(h["C0"] == h["C1"] for h in hashes.values()):\n'
            '        raise AssertionError("The width-64 draw must differ from the width-32 draw")\n',
            '        raise AssertionError("V14 C0 must reuse the V11 C1 per-seed parameter draw")\n'
            '    if any(h["C0"] != h["C1"] for h in hashes.values()):\n'
            '        raise AssertionError("Both V14 variants must share the V11 C1 per-seed draw")\n',
            1,
        ),
        (
            '            "only_difference_C0_C1": "C1 differs from C0 only by the carrier width: hidden_dim "\n'
            '            "64 and expansion_dim 128 instead of 32 and 64 (250542 instead of 64238 "\n'
            '            "parameters). "\n'
            '            f"The recipe is the {base_variant} one, fixed before any V11 result: TRAIN72, "\n',
            '            "only_difference_C0_C1": "C1 differs from C0 only by its frozen TRAIN set: "\n'
            '            f"TRAIN72 plus {len(ext_records)} TRAIN-EXT scenes instead of TRAIN72 (identical "\n'
            '            "architecture, 64238 parameters and per-seed draw). "\n'
            '            f"The recipe is the {base_variant} one: width 32, "\n',
            1,
        ),
        (
            '            f"+ {EXTRA_SEED_OFFSET}); scene order, query slots and ray indices are identical. "\n',
            '            f"+ {EXTRA_SEED_OFFSET}); the scene permutation covers the variant\'s own TRAIN "\n'
            '            "set, so C1\'s scene order, query slots and ray indices differ from C0\'s. "\n',
            1,
        ),
        (
            '        "primary_contrast": "EVAL-V4 WIDTH64_COMPLETION_GAIN = FAR AbsRel(V11 C1 selected, "\n'
            '        "the plan\'s control) - FAR AbsRel(C1 width 64); secondary HARMONIC_HYBRID_GAIN = "\n'
            '        "AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 of C1) and HFILL8_WIDTH_GAIN = "\n'
            '        "AbsRel(HFILL8 of V11 C1) - AbsRel(HFILL8 of C1)",\n'
            '        "secondary_contrast": "DEV WIDTH_GAIN_DEV = AbsRel(C0) - AbsRel(C1), frozen V2 "\n',
            '        "primary_contrast": "pooled EVAL-V3 + EVAL-V4 DATA_GAIN = FAR AbsRel(C0 TRAIN72) - "\n'
            '        "FAR AbsRel(C1 TRAIN72+EXT) (frozen V2 gate) and HARMONIC_HYBRID_GAIN = "\n'
            '        "AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 of C1) (INPAINT-V1 label rule)",\n'
            '        "secondary_contrast": "DEV DATA_GAIN_DEV = AbsRel(C0) - AbsRel(C1), frozen V2 "\n',
            1,
        ),
        (
            '        "base_basis": f"{PLAN} (sha256 {PLAN_SHA256}), written before any V11 result: "\n'
            '        f"V12 runs only if the V11 EVAL-V3 COMPLETION_GAIN point estimate is >0; "\n'
            '        f"COMPLETION_GAIN={gain!r}",\n',
            '        "base_basis": f"{PLAN} (sha256 {PLAN_SHA256}), written before any TRAIN-EXT scene "\n'
            '        "was read: the unchanged V11 C1 recipe, only the TRAIN set grows",\n',
            1,
        ),
        (
            '        "eval_v4": {\n'
            '            "manifest": str(eval_manifest.resolve()),\n'
            '            "manifest_sha256": sha(eval_manifest),\n'
            '            "scenes": len(eval_scenes),\n'
            '            "reprojection_fill_constant_m": v11_config["eval_v3"]["reprojection_fill_constant_m"],\n'
            '            "near_px": v11_config["eval_v3"]["near_px"],\n'
            '            "control": {\n'
            '                "selected": str(v11_selected.resolve()),\n'
            '                "selected_sha256": sha(v11_selected),\n'
            '                "method": "V11 C1 selected checkpoints, V11 loader, never retrained",\n'
            "            },\n"
            '            "role": "primary mechanism cohort: never read before its preparation, scene- "\n'
            '            "and asset-disjoint from every used scene and from EVAL-V3, volume-disjoint "\n'
            '            "from DEV and FRESH, may share volumes with TRAIN72 and EVAL-V3; never a "\n'
            '            "qualification cohort",\n'
            "        },\n"
            '        "train_sets": {"TRAIN24": train24, "TRAIN72": train72},\n',
            '        "eval": {\n'
            '            "cohorts": cohorts,\n'
            '            "primary": ["EVAL_V3", "EVAL_V4"],\n'
            '            "descriptive": ["EVAL_FAR"],\n'
            '            "reprojection_fill_constant_m": v11_config["eval_v3"]["reprojection_fill_constant_m"],\n'
            '            "near_px": v11_config["eval_v3"]["near_px"],\n'
            '            "abstain_opacity_threshold": 0.5,\n'
            '            "c0_reproduction_tolerance": 1e-6,\n'
            '            "role": "mechanism cohorts: scene- and asset-disjoint from TRAIN72 and "\n'
            '            "TRAIN-EXT, volume-disjoint from DEV, may share volumes with the training "\n'
            '            "scenes; EVAL-V3 and EVAL-V4 are disjoint and pooled for the primary "\n'
            '            "contrasts; EVAL-FAR reuses their scenes with far queries and is descriptive; "\n'
            '            "never a qualification cohort",\n'
            "        },\n"
            '        "train_ext": {\n'
            '            "manifest": str((TRAIN_EXT / "manifest_train_ext.json").resolve()),\n'
            '            "manifest_sha256": sha(TRAIN_EXT / "manifest_train_ext.json"),\n'
            '            "candidate_lock_sha256": sha(TRAIN_EXT / "candidate_lock.json"),\n'
            '            "scenes": len(ext_records),\n'
            "        },\n"
            '        "train_sets": {"TRAIN24": train24, "TRAIN72": train72, "TRAIN_EXT": train_ext},\n',
            1,
        ),
        (
            '        "sampling": "seeded scene permutation cycled; primary query slot cycles per scene visit; "\n'
            '        "uniform with-replacement rays, identical indices for A/B and all variants; the "\n'
            '        "extra-view count comes from its own generator so the ray stream is unchanged",\n'
            '        "initialize": "from scratch; torch.manual_seed(seed) then the width\'s carrier; C0 "\n'
            '        "equals the V11 C1 draw; the depth bypass starts at zero without consuming the RNG",\n',
            '        "sampling": "seeded permutation of the variant\'s own TRAIN set, cycled; primary query "\n'
            '        "slot cycles per scene visit; uniform with-replacement rays, identical indices for "\n'
            '        "A/B; the extra-view count comes from its own generator so the ray stream is "\n'
            '        "unchanged",\n'
            '        "initialize": "from scratch; torch.manual_seed(seed) then the width-32 carrier; C0 "\n'
            '        "and C1 equal the V11 C1 draw; the depth bypass starts at zero without consuming "\n'
            '        "the RNG",\n',
            1,
        ),
        (
            '        "execution": "all six runs (2 widths x 3 seeds) concurrently, one core each, on CPU "\n'
            '        "cores 8-23, possibly alongside V10 (its directories are excluded from the seal); "\n'
            '        "no GPU is used or queried (GPU paused by the user); min_free_gpu_mib is unused",\n',
            '        "execution": "all six runs (2 TRAIN sets x 3 seeds) concurrently, one core each, on "\n'
            '        "CPU cores 8-23; no GPU is used or queried (GPU paused by the user); "\n'
            '        "min_free_gpu_mib is unused",\n',
            1,
        ),
        (
            '        "denote the secondary DEV C0-C1 contrast, i.e. WIDTH_GAIN_DEV in V12",\n'
            '        "dev_reuse": "the 8 DEV scenes were evaluated in V2-V11 and Core B V1/V2; in V12 "\n'
            '        "DEV only selects checkpoints and gives descriptive results; the primary cohort is "\n'
            '        "EVAL-V4; FRESH-V1, FRESH-V2 and EVAL-V3 are not used",\n',
            '        "denote the secondary DEV C0-C1 contrast, i.e. DATA_GAIN_DEV in V14",\n'
            '        "dev_reuse": "the 8 DEV scenes were evaluated in V2-V12 and Core B V1/V2; in V14 "\n'
            '        "DEV only selects checkpoints and gives descriptive results; the primary cohort is "\n'
            '        "the pooled EVAL-V3 + EVAL-V4; FRESH-V1 and FRESH-V2 are not used",\n',
            1,
        ),
        (
            '            "authority": "mcss.mechanism_pilot.rgbd_width_carrier.training_loss: the V8 "\n',
            '            "authority": "mcss.mechanism_pilot.rgbd_data_scale_carrier.training_loss: the V8 "\n',
            1,
        ),
        ('            "WIDTH_DEV_STATUS": "secondary, DEV:', '            "DATA_DEV_STATUS": "secondary, DEV:', 1),
        (
            '            "EVAL_V4_PRIMARY": {\n'
            '                "WIDTH64_COMPLETION_STATUS": "frozen V2 gate checks (mean>0, CIlo>0, >=.75 "\n'
            '                "nonworse, all LOSO>0, positive top1 share<=.5) on WIDTH64_COMPLETION_GAIN = "\n'
            '                "FAR AbsRel(V11 C1) - FAR AbsRel(C1) over EVAL-V4 scenes",\n'
            '                "HARMONIC_HYBRID_LABEL": "INPAINT-V1 label rule (ABOVE / "\n'
            '                "NOT_DISTINGUISHABLE / BELOW) on AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 "\n'
            '                "of C1)",\n'
            '                "HFILL8_WIDTH_STATUS": "frozen V2 gate checks on AbsRel(HFILL8 of V11 C1) - "\n'
            '                "AbsRel(HFILL8 of C1), secondary",\n',
            '            "EVAL_PRIMARY": {\n'
            '                "DATA_STATUS": "frozen V2 gate checks (mean>0, CIlo>0, >=.75 "\n'
            '                "nonworse, all LOSO>0, positive top1 share<=.5) on DATA_GAIN = "\n'
            '                "FAR AbsRel(C0) - FAR AbsRel(C1) over the pooled EVAL-V3 + EVAL-V4 scenes",\n'
            '                "HARMONIC_HYBRID_LABEL": "INPAINT-V1 label rule (ABOVE / "\n'
            '                "NOT_DISTINGUISHABLE / BELOW) on AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 "\n'
            '                "of C1) over the same scenes",\n'
            '                "HFILL8_DATA_STATUS": "frozen V2 gate checks on AbsRel(HFILL8 of C0) - "\n'
            '                "AbsRel(HFILL8 of C1), secondary",\n'
            '                "AHFILL8": "HFILL8 whose FAR pixels of rendered opacity < 0.5 also take "\n'
            '                "REPROJ_HARMONIC (V13); descriptive, its threshold was chosen on EVAL-V3/V4",\n'
            '                "C0_REPRODUCES_V11_C1": "C0 predictions equal the sealed V11 C1 predictions "\n'
            '                "of every cohort (max abs <= 1e-6) before any query depth is read, else stop",\n',
            1,
        ),
        (
            '                "A": "WIDTH64_COMPLETION_STATUS SUPPORTED and HARMONIC_HYBRID_LABEL ABOVE on "\n'
            '                "EVAL-V4: completion keeps improving with width, and the widest carrier beats "\n'
            '                "classical inpainting inside the deployable hybrid",\n'
            '                "B": "exactly one of the two EVAL-V4 conditions holds",\n',
            '                "A": "DATA_STATUS SUPPORTED and HARMONIC_HYBRID_LABEL ABOVE on the pooled "\n'
            '                "EVAL-V3 + EVAL-V4: more training scenes improve completion, and the carrier "\n'
            '                "trained on them beats classical inpainting inside the deployable hybrid",\n'
            '                "B": "exactly one of the two conditions holds",\n',
            1,
        ),
        (
            '    source_paths += sorted(Path("scripts").glob("*rgbd_width*.py"))\n',
            '    source_paths += sorted(Path("scripts").glob("*rgbd_data_scale*.py"))\n',
            1,
        ),
        (
            '    print("FROZEN", counts, "previous_files_sealed", sealed)\n',
            '    sizes = {"TRAIN72": len(train72), "TRAIN_EXT": len(train_ext), "DEV": counts["DEV"]}\n'
            '    print("FROZEN", sizes, "previous_files_sealed", sealed)\n',
            1,
        ),
    ],
    (
        "rgbd_width",
        '"""V12',
        "width64",
        "WIDTH64",
        "width 64",
        "EVAL_V4 =",
        '"eval_v4"',
        "COMPLETION_GAIN",
        "gain!r",
        "WIDTH_",
        "V12 C0",
        "in V12",
    ),
)

# ---------------------------------------------------------------- finalize
derive(
    "scripts/finalize_rgbd_width_carrier.py",
    "scripts/finalize_rgbd_data_scale_carrier.py",
    [
        (
            '"""V12 independent postrun checks, preservation audit, and portable numeric archive."""',
            '"""V14 independent postrun checks, preservation audit, and portable numeric archive."""',
            1,
        ),
        (CARRIER[0] + "VARIANT_SPECS\n", CARRIER[1] + "VARIANT_SPECS\n", 1),
        ('if root.name != "EXP-3D-RGBD-WIDTH-V12"', 'if root.name != "EXP-3D-RGBD-DATA-SCALE-V14"', 1),
        (
            '            train = {r["scene_id"] for r in manifest["scenes"] if r["split"] == "TRAIN"}\n'
            '            if not all(r["scene_id"] in train and np.isfinite(r["loss"]) for r in logs):\n'
            '                raise AssertionError("Non-TRAIN or nonfinite training")\n',
            '            train = set(config["train_sets"][VARIANT_SPECS[variant]["train"]])\n'
            '            if not all(r["scene_id"] in train and np.isfinite(r["loss"]) for r in logs):\n'
            "                raise AssertionError(\"Training outside the variant's TRAIN set or nonfinite\")\n",
            1,
        ),
        (*STREAM_COMMENT, 1),
        (*TRAIN_SETS, 1),
        (
            "    # EVAL-V4 (primary): predictions sealed before any query depth; results reproduce from rows.\n"
            '    eval_dir = root / "raw" / "eval_v4"\n',
            "    # EVAL (pooled EVAL-V3 + EVAL-V4 primary, EVAL-FAR descriptive): predictions sealed before\n"
            "    # any query depth, C0 equals the sealed V11 C1; results reproduce from rows.\n"
            '    eval_dir = root / "raw" / "eval"\n',
            1,
        ),
        ('raise AssertionError("EVAL-V4 seal record inconsistent")', 'raise AssertionError("EVAL seal record inconsistent")', 1),
        ('raise AssertionError("EVAL-V4 sealed prediction changed")', 'raise AssertionError("EVAL sealed prediction changed")', 1),
        (
            '        raise AssertionError("Query depth read before the EVAL-V4 seal")\n',
            '        raise AssertionError("Query depth read before the EVAL seal")\n'
            '    tolerance = config["eval"]["c0_reproduction_tolerance"]\n'
            '    if not seal_file["consistency"]["c0_minus_v11_c1_max_abs"] <= tolerance:\n'
            '        raise AssertionError("V14 C0 does not reproduce the sealed V11 C1 predictions")\n',
            1,
        ),
        (
            '        "v12_eval_v4", Path("scripts/evaluate_rgbd_width_eval_v4.py")\n',
            '        "v14_eval", Path("scripts/evaluate_rgbd_data_scale_eval.py")\n',
            1,
        ),
        (
            '    saved = read(root / "eval_v4_results.json")\n'
            "    for key in (\n"
            '        "WIDTH64_COMPLETION_GAIN",\n'
            '        "HARMONIC_HYBRID_GAIN",\n'
            '        "HFILL8_WIDTH_GAIN",\n'
            '        "statuses",\n'
            '        "INTERPRETATION_BRANCH",\n'
            "    ):\n",
            '    saved = read(root / "eval_results.json")\n'
            '    for key in ("primary", "cohorts", "statuses", "INTERPRETATION_BRANCH"):\n',
            1,
        ),
        (
            '            raise AssertionError(f"EVAL-V4 result does not reproduce from raw rows: {key}")\n',
            '            raise AssertionError(f"EVAL result does not reproduce from raw rows: {key}")\n',
            1,
        ),
        ('            "scripts/report_rgbd_width_carrier.py",\n', '            "scripts/report_rgbd_data_scale_carrier.py",\n', 1),
        ('        Path("scripts").glob("*rgbd_width*.py")\n', '        Path("scripts").glob("*rgbd_data_scale*.py")\n', 1),
        (
            '    source += sorted(Path("tests").glob("test_rgbd_width*.py"))\n',
            '    source += sorted(Path("tests").glob("test_rgbd_data_scale*.py"))\n',
            1,
        ),
    ],
    (*LEFT, "V12", "eval_v4", "EVAL-V4 seal", "EVAL-V4 (primary)"),
)

# ---------------------------------------------------------------- runner
derive(
    "scripts/run_rgbd_width_experiment.py",
    "scripts/run_rgbd_data_scale_experiment.py",
    [
        (
            '"""Execute the frozen V12 experiment (CPU only) and audits, never opening fresh data.',
            '"""Execute the frozen V14 experiment (CPU only) and audits, never opening fresh data.',
            1,
        ),
        ('                    "scripts/train_rgbd_width_carrier.py",\n', '                    "scripts/train_rgbd_data_scale_carrier.py",\n', 1),
        (
            '            ["scripts/evaluate_rgbd_width_carrier.py", "--root", str(root), "--device", "cpu"],\n',
            "            [\n"
            '                "scripts/evaluate_rgbd_data_scale_carrier.py",\n'
            '                "--root",\n'
            "                str(root),\n"
            '                "--device",\n'
            '                "cpu",\n'
            "            ],\n",
            1,
        ),
        (
            '        ["scripts/diagnose_rgbd_width_states.py", "--root", str(root)],\n',
            '        ["scripts/diagnose_rgbd_data_scale_states.py", "--root", str(root)],\n',
            1,
        ),
        (
            '        ("geometry_free_reference", "scripts/reference_rgbd_width_carrier.py"),\n'
            '        ("analysis", "scripts/analyze_rgbd_width_carrier.py"),\n'
            '        ("statistics_audit", "scripts/audit_rgbd_width_statistics.py"),\n',
            '        ("geometry_free_reference", "scripts/reference_rgbd_data_scale_carrier.py"),\n'
            '        ("analysis", "scripts/analyze_rgbd_data_scale_carrier.py"),\n'
            '        ("statistics_audit", "scripts/audit_rgbd_data_scale_statistics.py"),\n',
            1,
        ),
        (
            "    # Primary EVAL-V4 stage: predictions sealed before query depth, then scored, analyzed.\n"
            "    for stage, done in (\n"
            '        ("seal", root / "raw/eval_v4/prediction_seal.json"),\n'
            '        ("score", root / "raw/eval_v4/query_rows.json"),\n'
            '        ("analyze", root / "eval_v4_results.json"),\n'
            "    ):\n",
            "    # Primary EVAL stage (EVAL-V3, EVAL-V4, EVAL-FAR): sealed before query depth, then scored.\n"
            "    for stage, done in (\n"
            '        ("seal", root / "raw/eval/prediction_seal.json"),\n'
            '        ("score", root / "raw/eval/query_rows.json"),\n'
            '        ("analyze", root / "eval_results.json"),\n'
            "    ):\n",
            1,
        ),
        ('                f"eval_v4_{stage}",\n', '                f"eval_{stage}",\n', 1),
        ('                    "scripts/evaluate_rgbd_width_eval_v4.py",\n', '                    "scripts/evaluate_rgbd_data_scale_eval.py",\n', 1),
        ('Path("/tmp/rgbd-width-v12-finalization.log")', 'Path("/tmp/rgbd-data-scale-v14-finalization.log")', 1),
        ('                "scripts/finalize_rgbd_width_carrier.py",\n', '                "scripts/finalize_rgbd_data_scale_carrier.py",\n', 1),
    ],
    (*LEFT, "V12", "eval_v4", "Primary EVAL-V4", "rgbd-width"),
)
print("V14 DERIVATION COMPLETE")
