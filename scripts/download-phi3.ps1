# Downloads the Phi-3 Mini 4K Instruct (ONNX/GenAI, DirectML INT4 AWQ) model
# into model/phi-3-mini-4k-instruct-onnx for use with the Workbench Summarizer tab.
#
# The model is served from HuggingFace:
#   microsoft/Phi-3-mini-4k-instruct-onnx (directml-int4-awq-block-128 subdirectory)
#
# Required files (~2.3 GB total):
#   config.json, tokenizer.json, model.onnx, model.onnx.data
#
# Usage:
#   pwsh scripts/download-phi3.ps1
#   pwsh scripts/download-phi3.ps1 -Force -Validate
[CmdletBinding()]
param(
    [string]$ModelRoot = (Join-Path $PSScriptRoot '..\model'),
    [switch]$Force,
    [switch]$Validate
)
$ErrorActionPreference = 'Stop'

# Load shared helper
. (Join-Path $PSScriptRoot '_Download-HfModel.ps1')

$Repo = 'microsoft/Phi-3-mini-4k-instruct-onnx'
$Subdir = 'directml-int4-awq-block-128'
$targetDir = Join-Path $ModelRoot 'phi-3-mini-4k-instruct-onnx'

Write-Host "Downloading Phi-3 Mini 4K Instruct (ONNX/GenAI, DirectML INT4)"
Write-Host "  Source: $Repo/$Subdir"
Write-Host "  Target: $targetDir"
Write-Host ""

# Create target directory
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
}
$targetDir = (Resolve-Path -LiteralPath $targetDir).Path

$base = "https://huggingface.co/$Repo/resolve/main/$Subdir"
$requiredFiles = @('model.onnx', 'model.onnx.data', 'tokenizer.json', 'config.json')

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
Write-Host "Phi-3 model ready at: $targetDir"
Write-Host "Use the Workbench Summarizer tab or:"
Write-Host "  -Dphi3.modelDir=`"$targetDir`""
