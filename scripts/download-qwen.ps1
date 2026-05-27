# Downloads the Qwen2.5-Coder 0.5B Instruct (ONNX INT4 AWQ block-128) model
# into model/qwen2.5-coder-0.5b-directml-int4 for use with the Workbench.
#
# STATUS: PLANNED / SOURCE TBD
#
# The concrete HuggingFace ONNX source is TBD/research. The candidate under
# evaluation is onnx-community/Qwen2.5-Coder-0.5B-Instruct but its layout
# compatibility with DirectML INT4 AWQ block-128 has not been verified yet.
# See docs/decision-qwen-artifact-format.md for background.
#
# Required files (estimated ~350 MB total):
#   config.json, tokenizer.json, tokenizer_config.json,
#   special_tokens_map.json, model.onnx, model.onnx.data
#
# Optional files:
#   added_tokens.json
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

# Load shared helper
. (Join-Path $PSScriptRoot '_Download-HfModel.ps1')

# --- Source configuration (TBD/research) ---
# Update this once source verification completes (see #96).
$Repo = 'onnx-community/Qwen2.5-Coder-0.5B-Instruct'
$Subdir = 'directml/directml-int4-awq-block-128'
$targetDir = Join-Path $ModelRoot 'qwen2.5-coder-0.5b-directml-int4'

Write-Host "Downloading Qwen2.5-Coder 0.5B Instruct (ONNX INT4 AWQ block-128)"
Write-Host "  Source: $Repo/$Subdir"
Write-Host "  Target: $targetDir"
Write-Host ""
Write-Host "  WARNING: ONNX source is TBD/research. Download may fail if the"
Write-Host "  source repository or subdirectory layout is not yet available."
Write-Host ""

# Create target directory
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
}
$targetDir = (Resolve-Path -LiteralPath $targetDir).Path

$base = "https://huggingface.co/$Repo/resolve/main/$Subdir"
$requiredFiles = @('model.onnx', 'model.onnx.data', 'tokenizer.json', 'config.json',
                   'tokenizer_config.json', 'special_tokens_map.json')
$optionalFiles = @('added_tokens.json')

foreach ($f in $requiredFiles) {
    $dst = Join-Path $targetDir $f
    if ((Test-Path $dst) -and -not $Force) {
        Write-Host "  skip $f (exists, use -Force to re-download)"
        continue
    }
    Write-Host "  downloading $f ..."
    try {
        Invoke-WebRequest -Uri "$base/$f" -OutFile $dst -UseBasicParsing
    } catch {
        throw "Failed to download required file '$f' from $Repo/$Subdir : $($_.Exception.Message)"
    }
}

foreach ($f in $optionalFiles) {
    $dst = Join-Path $targetDir $f
    if ((Test-Path $dst) -and -not $Force) {
        Write-Host "  skip $f (exists)"
        continue
    }
    Write-Host "  downloading (optional) $f ..."
    try {
        Invoke-WebRequest -Uri "$base/$f" -OutFile $dst -UseBasicParsing
    } catch {
        Write-Host "  optional file not found (skipped): $f"
    }
}

# Validate
if ($Validate) {
    Write-Host ""
    Write-Host "  File validation:"
    $missing = @()
    foreach ($f in $requiredFiles) {
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
}

Write-Host ""
Write-Host "Qwen2.5-Coder 0.5B model ready at: $targetDir"
Write-Host ""
Write-Host "NOTE: Runtime support is not yet available. The model cannot be"
Write-Host "used for text generation until runtime implementation (#99) is complete."
