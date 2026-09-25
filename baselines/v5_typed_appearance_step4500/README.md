# V5 Typed-Appearance Baseline

## Status

This bundle freezes the current research comparison anchor:

- branch: V5 typed high-resolution appearance state;
- checkpoint: step 4500;
- seed: 17;
- dataset role: one-window Hypersim development-train overfit;
- architecture: `evidence_residual`;
- appearance resolution scale: 2;
- renderer: fixed and parameter-free.

"Frozen" means recoverable and never forgotten. It does not mean future code cannot change.

## Why this version

V5 step 4500 is the strongest accepted engineering/capacity reference retained by the later V6
capacity gate. V6a and V6a.1 did not replace it: V6a.1 kept RGB within its gate but worsened depth
AbsRel by 11.07% and normal angle by 7.00%, so that branch was sealed. V7 and V8 are supplier or
risk audits and are not trained replacements for the typed-state model.

The historical A+ validation/test runs are broader evaluations but have much lower RGB scores and
belong to an earlier architecture. The V5 multi-scene pilot validation did not establish a new
best model because geometry collapsed. Therefore, V5 step 4500 is retained as the engineering
anchor while its lack of a clean multi-scene validation result remains explicit.

## Reference metrics

These metrics come from one training window and must not be described as validation or test
performance:

| Metric | Value |
| --- | ---: |
| RGB PSNR | 21.065804 dB |
| RGB SSIM | 0.760582 |
| Depth AbsRel | 0.00889812 |
| Depth RMSE | 0.059240 m |
| Normal mean angle | 6.499368 deg |
| Point F-score at 0.25 m | 0.994383 |
| Point MAE | 0.017288 m |
| Support coverage | 1.0 |

## Bundle contents

- `checkpoint/step_004500.pt`: frozen checkpoint copy.
- `config/hypersim_er_v5_appearance2x_overfit.yaml`: authored run config.
- `config/resolved_config.json`: fully resolved run config.
- `evidence/evaluation.json`: compact metric artifact.
- `evidence/evaluation_report.json`: metric definitions, provenance, and checkpoint hash.
- `evidence/train.jsonl`: complete training log from the V5 run.
- `source/compatible_source_20260813.zip`: source/config/test/docs snapshot known to load the V5
  checkpoint. It also preserves later independent audit modules; the V5 architecture is selected by
  the frozen config.
- `MANIFEST.sha256`: integrity hashes for every frozen artifact.

## Restoration and use

The current working tree can load the frozen checkpoint directly:

```powershell
cd <workspace-root>
uv run mcss evaluate `
  --config baselines\v5_typed_appearance_step4500\config\hypersim_er_v5_appearance2x_overfit.yaml `
  --checkpoint baselines\v5_typed_appearance_step4500\checkpoint\step_004500.pt `
  --output-dir outputs\v5_baseline_recheck
```

Do not reuse an existing evaluation output directory because the evaluator refuses to overwrite
metric artifacts. Extract `source/compatible_source_20260813.zip` into a separate recovery folder
only if the current working code later stops loading this checkpoint.

## Replacement rule

A future candidate does not replace this pointer merely by producing a better training-window
number. The comparison protocol, data split, metrics, and acceptance thresholds must be declared
before opening candidate results. At minimum, compare the candidate and V5 on the same untouched
windows and report RGB, geometry, support, uncertainty, and fixed-renderer contract checks. Until
that gate passes, V5 remains the baseline and the candidate remains an experiment branch.
