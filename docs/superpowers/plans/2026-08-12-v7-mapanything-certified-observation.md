# V7 MapAnything-Certified Observation Implementation Plan

> Execute in the existing non-Git project directory. Preserve V5/V6 and the sealed LoFTR audit.
> Every production interface follows RED-GREEN-REFACTOR; no state integration or training occurs.

## File responsibilities

- `src/mcss/geometry_supplier.py`: supplier request/result contracts, canonical ordering, hashes,
  and atomic structured cache.
- `src/mcss/mapanything_supplier.py`: lazy adapter to the pinned external model; no labels and no
  predicted-camera trust.
- `src/mcss/dense_geometry_certificates.py`: frozen known-camera consistency checks and typed ray
  observations.
- `src/mcss/dense_geometry_audit.py`: frozen-window runner, post-seal GT evaluation, controls,
  gates, reports, and resumability.
- `scripts/run_mapanything_geometry_audit.py`: thin CLI.
- `configs/hypersim_mapanything_geometry_audit.yaml`: the complete frozen protocol.
- `tests/test_geometry_supplier.py`: contract, canonicalization, hash, and atomic-cache tests.
- `tests/test_mapanything_supplier.py`: adapter boundary tests using an injected fake backend.
- `tests/test_dense_geometry_certificates.py`: analytic geometry/support/conflict/risk tests.
- `tests/test_dense_geometry_audit.py`: label-order, fixed-window, control, and decision tests.

## Task 1: supplier and cache contract

1. Write failing tests for request validation, deterministic frame sorting, label-free signatures,
   non-overwrite behavior, NPZ/JSON round-trip, and tamper detection.
2. Run the focused tests and confirm they fail because the module is absent.
3. Implement the smallest typed contract and atomic cache that satisfies them.
4. Run the focused tests and Ruff for the new files.

## Task 2: known-camera certificate builder

1. Write analytic failing tests using two/three pinhole cameras with consistent, occluded, and
   conflicting depth maps.
2. Confirm RED for missing certificate behavior.
3. Implement vectorized unprojection, projection, bilinear depth sampling, support/conflict logic,
   typed intervals, provenance, and fixed risk.
4. Verify invariance after frame-ID remapping and pass the focused suite.

## Task 3: audit evaluator and gates

1. Write failing tests showing certificates are serialized before any depth loader is called,
   raw confidence is compared at matched count, every gate is mandatory, and forbidden partitions
   are rejected.
2. Implement config parsing, exact frozen-window loading, sealed-cache workflow, post-hoc source
   error, shuffles, aggregation, report rendering, and artifact hashes.
3. Run focused tests with a deterministic fake supplier and inspect the generated audit tree.

## Task 4: pinned MapAnything adapter

1. Write failing injected-backend tests that assert the adapter sends only RGB, known intrinsics,
   known cam2world, and metric-scale flags, canonically ordered.
2. Implement lazy imports, official preprocessing, pinned `from_pretrained`, BF16 memory-efficient
   inference, and conversion to the generic supplier result.
3. Reject output shape/key/nonfinite violations and ignore predicted cameras/world points.

## Task 5: real supplier qualification run

1. Verify the external repository commit, DINOv2 architecture-code dependency, and model
   revision; record source/model hashes, licenses, and disk headroom.
2. Install the pinned code and download the pinned model without exposing credentials in logs.
3. Run one real four-view GPU smoke window into a new output directory; inspect cache hashes,
   CUDA peak memory, masks, depths, and label access order.
4. If the smoke fails mechanically, fix only implementation defects with regression tests. Do not
   change scientific thresholds based on GT.
5. Run the exact 32-window audit. Resume only from hash-validated sealed supplier/certificate
   caches.
6. Produce `audit_result.json`, `window_metrics.jsonl`, `access_log.json`, `REPORT.md`, and
   `artifact_hashes.json`; issue GO, STOP, or INCONCLUSIVE from the frozen gates.

## Task 6: completion verification

Run fresh:

```powershell
uv run --extra dev pytest -q
uv run --extra dev ruff check src scripts tests
python -m compileall -q src scripts tests
```

Also verify model/code revisions, output hashes, no diagnostic/final access, D-drive free space
above 100 GiB, and that production state/training files remain untouched. Report scientific
status separately from software verification.
