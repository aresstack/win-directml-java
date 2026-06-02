# Downloads Qwen2.5-Coder 0.5B Instruct ONNX files for the Workbench.
#
# The selected ONNX file keeps its original Hugging Face filename, e.g.
#   onnx/model_q4f16.onnx -> model_q4f16.onnx
#
# Usage:
#   pwsh scripts/download-qwen.ps1
#   pwsh scripts/download-qwen.ps1 -OnnxFile model_q4f16.onnx -Force -Validate
#   pwsh scripts/download-qwen.ps1 -AllOnnxVariants
[CmdletBinding()]
param(
    [string]$ModelRoot = (Join-Path $PSScriptRoot '..\model'),

    [ValidateSet('model.onnx', 'model_fp16.onnx', 'model_int8.onnx', 'model_uint8.onnx', 'model_quantized.onnx', 'model_q4.onnx', 'model_q4f16.onnx', 'model_bnb4.onnx')]
    [string]$OnnxFile = 'model_q4f16.onnx',

    [switch]$AllOnnxVariants,
    [switch]$Force,
    [switch]$Validate
)
$ErrorActionPreference = 'Stop'

$Repo = 'onnx-community/Qwen2.5-Coder-0.5B-Instruct'
$OnnxSubdir = 'onnx'
$targetDir = Join-Path $ModelRoot 'qwen2.5-coder-0.5b-directml-int4'
$knownOnnxFiles = @(
    'model.onnx',
    'model_fp16.onnx',
    'model_int8.onnx',
    'model_uint8.onnx',
    'model_quantized.onnx',
    'model_q4.onnx',
    'model_q4f16.onnx',
    'model_bnb4.onnx'
)

Write-Host 'Downloading Qwen2.5-Coder 0.5B Instruct'
Write-Host "  Source repo:   $Repo"
Write-Host "  ONNX subdir:   $OnnxSubdir"
Write-Host "  Target:        $targetDir"
Write-Host "  Selected ONNX: $OnnxFile"
Write-Host "  Download all:  $AllOnnxVariants"
Write-Host ''

if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
}
$targetDir = (Resolve-Path -LiteralPath $targetDir).Path

$hfBase = "https://huggingface.co/$Repo/resolve/main"
$onnxBase = "$hfBase/$OnnxSubdir"
$selectedModelUrl = "$onnxBase/$OnnxFile"

Write-Host "Selected model URL: $selectedModelUrl"
Write-Host ''

function Download-RequiredFile {
    param(
        [string]$Url,
        [string]$LocalName
    )
    $dst = Join-Path $targetDir $LocalName
    if ((Test-Path $dst) -and -not $Force) {
        Write-Host "  skip $LocalName (exists, use -Force to re-download)"
        return
    }
    Write-Host "  downloading $LocalName ..."
    try {
        Invoke-WebRequest -Uri $Url -OutFile $dst -UseBasicParsing
    } catch {
        throw "Failed to download required file '$LocalName' from $Url : $($_.Exception.Message)"
    }
}

function Download-OptionalFile {
    param(
        [string]$Url,
        [string]$LocalName
    )
    $dst = Join-Path $targetDir $LocalName
    if ((Test-Path $dst) -and -not $Force) {
        Write-Host "  skip $LocalName (exists)"
        return
    }
    Write-Host "  downloading (optional) $LocalName ..."
    try {
        Invoke-WebRequest -Uri $Url -OutFile $dst -UseBasicParsing
    } catch {
        Write-Host "  optional file not found (skipped): $LocalName"
    }
}

$onnxFilesToDownload = if ($AllOnnxVariants) { $knownOnnxFiles } else { @($OnnxFile) }
foreach ($file in $onnxFilesToDownload) {
    Download-RequiredFile -Url "$onnxBase/$file" -LocalName $file
    if ($file -eq 'model.onnx') {
        Download-RequiredFile -Url "$onnxBase/model.onnx_data" -LocalName 'model.onnx_data'
    }
}

$rootRequired = @('tokenizer.json', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json')
$rootOptional = @('added_tokens.json', 'generation_config.json')

foreach ($f in $rootRequired) {
    Download-RequiredFile -Url "$hfBase/$f" -LocalName $f
}

foreach ($f in $rootOptional) {
    Download-OptionalFile -Url "$hfBase/$f" -LocalName $f
}

if ($Validate) {
    Write-Host ''
    Write-Host '  File validation:'
    $allRequired = @($onnxFilesToDownload) + $rootRequired
    if ($onnxFilesToDownload -contains 'model.onnx') {
        $allRequired += 'model.onnx_data'
    }
    $missing = @()
    foreach ($f in $allRequired) {
        $dst = Join-Path $targetDir $f
        if (Test-Path $dst) {
            $size = (Get-Item $dst).Length
            $sizeStr = if ($size -gt 1MB) { '{0:N1} MB' -f ($size / 1MB) }
                       elseif ($size -gt 1KB) { '{0:N1} KB' -f ($size / 1KB) }
                       else { "$size B" }
            $hash = (Get-FileHash -Path $dst -Algorithm SHA256).Hash.Substring(0, 12)
            Write-Host ('    {0,-30} {1,10}  sha256:{2}' -f $f, $sizeStr, $hash)
        } else {
            $missing += $f
        }
    }
    if ($missing.Count -gt 0) {
        throw "Required files missing after download: $($missing -join ', ')"
    }
}

Write-Host ''
Write-Host "Qwen2.5-Coder 0.5B files ready at: $targetDir"
Write-Host "Selected ONNX file for Workbench runtime: $OnnxFile"
