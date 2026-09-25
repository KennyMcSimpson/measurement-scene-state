param(
    [Parameter(Mandatory = $true)]
    [string]$ExternalDataRoot,
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
if (-not $Execute) {
    throw 'This historical cleanup script is guarded. Re-run with -Execute after reviewing the exact inventory.'
}
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).ProviderPath
$externalRoot = (Resolve-Path -LiteralPath $ExternalDataRoot).ProviderPath
$datasetCleanupTargets = @(
    @{ Path = (Join-Path $projectRoot 'data/hypersim_er_prepared'); Files = 125870L; Bytes = 14960209419L },
    @{ Path = (Join-Path $projectRoot 'data/hypersim_final_holdout_archives'); Files = 26L; Bytes = 102405935184L },
    @{ Path = (Join-Path $projectRoot 'data/hypersim_final_holdout_raw'); Files = 136795L; Bytes = 102370486124L },
    @{ Path = (Join-Path $projectRoot 'data/hypersim_prepared'); Files = 29599L; Bytes = 3534935427L },
    @{ Path = (Join-Path $projectRoot 'data/hypersim_raw'); Files = 128027L; Bytes = 59103934584L },
    @{ Path = (Join-Path $externalRoot 'DL3DV_Benchmark_20260920'); Files = 13058L; Bytes = 11559355642L }
)
$datasetCleanupOutput = Join-Path $projectRoot 'outputs/dataset_cleanup_20260921'
New-Item -ItemType Directory -Path $datasetCleanupOutput -Force | Out-Null
$datasetCleanupReceipt = [ordered]@{
    authorization = 'User 2026-09-21: delete local old 3D datasets and start 2D discovery experiments'
    started_utc = [DateTime]::UtcNow.ToString('o')
    classification = 'Previously downloaded or reproducibly prepared CV dataset assets; no code/results/checkpoints'
    targets = @()
    status = 'checking'
}
$datasetCleanupReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $datasetCleanupOutput 'receipt.json') -Encoding UTF8
foreach ($datasetCleanupTarget in $datasetCleanupTargets) {
    $datasetCleanupItem = Get-Item -LiteralPath $datasetCleanupTarget.Path -Force
    $datasetCleanupResolved = (Resolve-Path -LiteralPath $datasetCleanupTarget.Path).ProviderPath
    if (-not $datasetCleanupItem.PSIsContainer) { throw "Not a directory: $datasetCleanupResolved" }
    if (($datasetCleanupItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Root reparse point: $datasetCleanupResolved" }
    $datasetCleanupScan = Get-ChildItem -LiteralPath $datasetCleanupResolved -Recurse -Force
    if (@($datasetCleanupScan | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 }).Count -ne 0) { throw "Nested reparse point: $datasetCleanupResolved" }
    $datasetCleanupFiles = @($datasetCleanupScan | Where-Object { -not $_.PSIsContainer })
    $datasetCleanupBytes = ($datasetCleanupFiles | Measure-Object -Property Length -Sum).Sum
    if ($datasetCleanupFiles.Count -ne $datasetCleanupTarget.Files -or $datasetCleanupBytes -ne $datasetCleanupTarget.Bytes) { throw "Inventory changed: $datasetCleanupResolved" }
    $datasetCleanupReceipt.targets += [ordered]@{ path = $datasetCleanupResolved; files = $datasetCleanupFiles.Count; bytes = $datasetCleanupBytes; reparse_count = 0; status = 'checked' }
    Write-Output "CHECKED $datasetCleanupResolved $($datasetCleanupFiles.Count) files $datasetCleanupBytes bytes"
}
$datasetCleanupActive = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|pythonw|aria2c|curl)\.exe$' -and $_.CommandLine -match 'download_hypersim|run_dl3dv|evaluate_dl3dv|train_dynamic|evaluate_dynamic' })
if ($datasetCleanupActive.Count -gt 0) { throw 'Old dataset-consuming process appeared; refusing cleanup' }
$datasetCleanupReceipt.status = 'deleting'
$datasetCleanupReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $datasetCleanupOutput 'receipt.json') -Encoding UTF8
foreach ($datasetCleanupEntry in $datasetCleanupReceipt.targets) {
    $datasetCleanupNow = Get-Item -LiteralPath $datasetCleanupEntry.path -Force
    if (($datasetCleanupNow.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Target changed into reparse point' }
    Remove-Item -LiteralPath $datasetCleanupEntry.path -Recurse -Force
    if (Test-Path -LiteralPath $datasetCleanupEntry.path) { throw "Deletion incomplete: $($datasetCleanupEntry.path)" }
    $datasetCleanupEntry.status = 'deleted_and_absent'
    Write-Output "DELETED $($datasetCleanupEntry.path)"
    $datasetCleanupReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $datasetCleanupOutput 'receipt.json') -Encoding UTF8
}
$datasetCleanupReceipt.status = 'complete'
$datasetCleanupReceipt.completed_utc = [DateTime]::UtcNow.ToString('o')
$datasetCleanupReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $datasetCleanupOutput 'receipt.json') -Encoding UTF8
