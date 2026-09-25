$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run mcss train --config configs\synthetic_smoke.yaml
uv run mcss evaluate `
  --config configs\synthetic_smoke.yaml `
  --checkpoint outputs\synthetic_smoke\checkpoints\step_000008.pt

