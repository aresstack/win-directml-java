# Downloads Qwen2.5-Coder 0.5B Instruct ONNX files from Hugging Face
# into model/qwen2.5-coder-0.5b-directml-int4 for use with the Workbench.
#
# The script keeps the original Hugging Face filenames. Runtime selection is
# stored in qwen-selected-onnx.txt, the same selection file used by the Workbench.
#
# Usage:
#   pwsh scripts/download-qwen.ps1
#   pwsh scripts/download-qwen.ps1 -Variant model_q4f16.onnx -Force -Validate
#   pwsh scripts/download-qwen.ps1 -AllVariants -Variant model_q4f16.onnx
[CmdletBinding()]
param(
    [string]$ModelRoot = (Join-Path $PSScriptRoot '..\model'),
    [ValidateSet('model_q4f16.onnx','model_q4.onnx','model_bnb4.onnx','model_int8.onnx','model_uint8.onnx','model_quantized.onnx','model_fp16.onnx','model.onnx')]
    [string]$Variant = 'model_q4f16.onnx',
    [switch]$AllVariants,
    [switch]$Force,
    [switch]$Validate
)
$ErrorActionPreference = 'Stop'

$Repo = 'onnx-community/Qwen2.5-Coder-0.5B-Instruct'
$OnnxSubdir = 'onnx'
$targetDir = Join-Path $ModelRoot 'qwen2.5-coder-0.5b-directml-int4'
$selectionFile = 'qwen-selected-onnx.txt'

$knownOnnxFiles = @(
    'model_q4f16.onnx',
    'model_q4.onnx',
    'model_bnb4.onnx',
    'model_int8.onnx',
    'model_uint8.onnx',
    'model_quantized.onnx',
    'model_fp16.onnx',
    'model.onnx'
)

Write-Host 'Downloading Qwen2.5-Coder 0.5B Instruct'
Write-Host "  Source repo:     $Repo"
Write-Host "  ONNX subdir:     $OnnxSubdir"
Write-Host "  Selected ONNX:   $Variant"
Write-Host "  Download scope:  $(if ($AllVariants) { 'all ONNX files' } else { 'selected ONNX file only' })"
Write-Host "  Target:          $targetDir"
Write-Host ''

if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
}
$targetDir = (Resolve-Path -LiteralPath $targetDir).Path

$hfBase = "https://huggingface.co/$Repo/resolve/main"
$onnxBase = "$hfBase/$OnnxSubdir"
$onnxFilesToDownload = if ($AllVariants) { $knownOnnxFiles } else { @($Variant) }

function Download-File {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$LocalName,
        [Parameter(Mandatory=$true)][bool]$Required
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
        if ($Required) {
            throw "Failed to download required file '$LocalName' from $Url : $($_.Exception.Message)"
        }
        Write-Host "  optional file not found (skipped): $LocalName"
    }
}

foreach ($f in $onnxFilesToDownload) {
    Download-File -Url "$onnxBase/$f" -LocalName $f -Required $true
}

if ($AllVariants -or $Variant -eq 'model.onnx') {
    Download-File -Url "$onnxBase/model.onnx_data" -LocalName 'model.onnx_data' -Required $true
}

$rootRequired = @('tokenizer.json', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json')
$rootOptional = @('added_tokens.json', 'generation_config.json')

foreach ($f in $rootRequired) {
    Download-File -Url "$hfBase/$f" -LocalName $f -Required $true
}

foreach ($f in $rootOptional) {
    Download-File -Url "$hfBase/$f" -LocalName $f -Required $false
}

Set-Content -LiteralPath (Join-Path $targetDir $selectionFile) -Value $Variant -Encoding UTF8
Write-Host "  active ONNX selection: $Variant"
Write-Host "  selected URL: $onnxBase/$Variant"

if ($Validate) {
    Write-Host ''
    Write-Host '  File validation:'
    $allRequired = @($Variant) + $rootRequired
    if ($Variant -eq 'model.onnx') {
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
Write-Host "Qwen2.5-Coder 0.5B model ready at: $targetDir"
Write-Host "Runtime will use: $Variant"
