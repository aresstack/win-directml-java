# Downloads the Qwen2.5-Coder 0.5B Instruct (ONNX) model
# into model/qwen2.5-coder-0.5b-directml-int4 for use with the Workbench.
#
# Source: onnx-community/Qwen2.5-Coder-0.5B-Instruct (onnx/ subdir)
# Layout matches QwenModelDownloadConfig.DEFAULT in the Java workbench:
#   onnx/model.onnx         → model.onnx          (ONNX graph)
#   onnx/model.onnx_data    → model.onnx.data      (external weights, ~350 MB)
#   tokenizer.json          → tokenizer.json       (from repo root)
#   config.json             → config.json          (from repo root)
#   tokenizer_config.json   → tokenizer_config.json(from repo root)
#   special_tokens_map.json → special_tokens_map.json (from repo root)
#   added_tokens.json       → added_tokens.json    (optional, from repo root)
#
# Usage:
#   pwsh scripts/download-qwen.ps1
#   pwsh scripts/download-qwen.ps1 -Force -Validate
[CmdletBinding()]
param(
    [string]$ModelRoot = (Join-Path $PSScriptRoot '..\model'),
    [switch]$Force,
    [switch]$Validate
)
$ErrorActionPreference = 'Stop'

# --- Source configuration (matches QwenModelDownloadConfig.DEFAULT) ---
$Repo    = 'onnx-community/Qwen2.5-Coder-0.5B-Instruct'
$OnnxSubdir = 'onnx'
$targetDir  = Join-Path $ModelRoot 'qwen2.5-coder-0.5b-directml-int4'

Write-Host "Downloading Qwen2.5-Coder 0.5B Instruct"
Write-Host "  Source repo:   $Repo"
Write-Host "  ONNX subdir:   $OnnxSubdir"
Write-Host "  Target:        $targetDir"
Write-Host ""

# Create target directory
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
}
$targetDir = (Resolve-Path -LiteralPath $targetDir).Path

$hfBase      = "https://huggingface.co/$Repo/resolve/main"
$onnxBase    = "$hfBase/$OnnxSubdir"

# --- ONNX model files (from the onnx/ subdir) ---
$onnxFiles = @(
    @{ Remote = 'model.onnx';      Local = 'model.onnx' },
    @{ Remote = 'model.onnx_data'; Local = 'model.onnx.data' }
)

foreach ($entry in $onnxFiles) {
    $dst = Join-Path $targetDir $entry.Local
    if ((Test-Path $dst) -and -not $Force) {
        Write-Host "  skip $($entry.Local) (exists, use -Force to re-download)"
        continue
    }
    Write-Host "  downloading $($entry.Local) ..."
    try {
        Invoke-WebRequest -Uri "$onnxBase/$($entry.Remote)" -OutFile $dst -UseBasicParsing
    } catch {
        throw "Failed to download required file '$($entry.Local)' from $Repo/$OnnxSubdir : $($_.Exception.Message)"
    }
}

# --- Config / tokenizer files (from repo root) ---
$rootRequired = @('tokenizer.json', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json')
$rootOptional = @('added_tokens.json')

foreach ($f in $rootRequired) {
    $dst = Join-Path $targetDir $f
    if ((Test-Path $dst) -and -not $Force) {
        Write-Host "  skip $f (exists, use -Force to re-download)"
        continue
    }
    Write-Host "  downloading $f ..."
    try {
        Invoke-WebRequest -Uri "$hfBase/$f" -OutFile $dst -UseBasicParsing
    } catch {
        throw "Failed to download required file '$f' from $Repo root: $($_.Exception.Message)"
    }
}

foreach ($f in $rootOptional) {
    $dst = Join-Path $targetDir $f
    if ((Test-Path $dst) -and -not $Force) {
        Write-Host "  skip $f (exists)"
        continue
    }
    Write-Host "  downloading (optional) $f ..."
    try {
        Invoke-WebRequest -Uri "$hfBase/$f" -OutFile $dst -UseBasicParsing
    } catch {
        Write-Host "  optional file not found (skipped): $f"
    }
}

# --- Validate ---
if ($Validate) {
    Write-Host ""
    Write-Host "  File validation:"
    $allRequired = @('model.onnx') + $rootRequired
    $missing = @()
    foreach ($f in $allRequired) {
        $dst = Join-Path $targetDir $f
        if (Test-Path $dst) {
            $size = (Get-Item $dst).Length
            $sizeStr = if ($size -gt 1MB) { "{0:N1} MB" -f ($size / 1MB) }
                       elseif ($size -gt 1KB) { "{0:N1} KB" -f ($size / 1KB) }
                       else { "$size B" }
            $hash = (Get-FileHash -Path $dst -Algorithm SHA256).Hash.Substring(0, 12)
            Write-Host ("    {0,-30} {1,10}  sha256:{2}" -f $f, $sizeStr, $hash)
        } else {
            $missing += $f
        }
    }
    if ($missing.Count -gt 0) {
        throw "Required files missing after download: $($missing -join ', ')"
    }

    $primaryData = Join-Path $targetDir 'model.onnx.data'
    $legacyData = Join-Path $targetDir 'model.onnx_data'
    if (Test-Path $primaryData) {
        Write-Host "    external data file              model.onnx.data (preferred)"
    } elseif (Test-Path $legacyData) {
        Write-Host "    external data file              model.onnx_data (legacy fallback)"
    } else {
        throw "Required external data file missing after download: model.onnx.data (or legacy model.onnx_data)"
    }
}

Write-Host ""
Write-Host "Qwen2.5-Coder 0.5B model ready at: $targetDir"
Write-Host ""
Write-Host "NOTE: Workbench exposes Qwen as a local CPU test path."
Write-Host "The model status remains 'planned' until source/layout verification"
Write-Host "and a real end-to-end generation smoke test pass (see docs/qwen-smoke-test.md)."
