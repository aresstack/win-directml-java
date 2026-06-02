# Downloads the Qwen2.5-Coder 0.5B Instruct q4f16 ONNX model
# into model/qwen2.5-coder-0.5b-directml-int4 for use with the Workbench.
#
# Source: onnx-community/Qwen2.5-Coder-0.5B-Instruct
# Layout:
#   onnx/model_q4f16.onnx → model.onnx          (single-file quantized ONNX)
#   tokenizer.json        → tokenizer.json       (from repo root)
#   config.json           → config.json          (from repo root)
#   tokenizer_config.json → tokenizer_config.json(from repo root)
#   special_tokens_map.json → special_tokens_map.json (from repo root)
#   added_tokens.json     → added_tokens.json    (optional, from repo root)
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

$Repo = 'onnx-community/Qwen2.5-Coder-0.5B-Instruct'
$TargetDir = Join-Path $ModelRoot 'qwen2.5-coder-0.5b-directml-int4'
$HfBase = "https://huggingface.co/$Repo/resolve/main"
$ExpectedArtifact = 'onnx/model_q4f16.onnx'

Write-Host "Downloading Qwen2.5-Coder 0.5B Instruct q4f16"
Write-Host "  Source repo:   $Repo"
Write-Host "  Artifact:      $ExpectedArtifact"
Write-Host "  Target:        $TargetDir"
Write-Host ""

if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
}
$TargetDir = (Resolve-Path -LiteralPath $TargetDir).Path

function Get-ExistingArtifactName {
    param([string]$Directory)
    $manifest = Join-Path $Directory 'model-source.properties'
    if (-not (Test-Path $manifest)) {
        return $null
    }
    foreach ($line in Get-Content -LiteralPath $manifest) {
        if ($line -match '^artifact=(.*)$') {
            return $Matches[1].Trim()
        }
    }
    return $null
}

function Download-File {
    param(
        [string]$RemotePath,
        [string]$LocalName,
        [bool]$Required,
        [bool]$ForceDownload
    )

    $destination = Join-Path $TargetDir $LocalName
    if ((Test-Path $destination) -and -not $ForceDownload) {
        Write-Host "  skip $LocalName (exists, use -Force to re-download)"
        return
    }

    $temporary = "$destination.tmp"
    if (Test-Path $temporary) {
        Remove-Item -LiteralPath $temporary -Force
    }

    $url = "$HfBase/$RemotePath"
    Write-Host "  downloading $LocalName ..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $temporary -UseBasicParsing
        Move-Item -LiteralPath $temporary -Destination $destination -Force
    } catch {
        if (Test-Path $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
        if ($Required) {
            throw "Failed to download required file '$LocalName' from $url : $($_.Exception.Message)"
        }
        Write-Host "  optional file not found (skipped): $LocalName"
    }
}

$removedStaleSidecar = $false
foreach ($stale in @('model.onnx_data', 'model.onnx.data')) {
    $stalePath = Join-Path $TargetDir $stale
    if (Test-Path $stalePath) {
        Write-Host "  removing stale dense sidecar $stale"
        Remove-Item -LiteralPath $stalePath -Force
        $removedStaleSidecar = $true
    }
}

$currentArtifact = Get-ExistingArtifactName -Directory $TargetDir
$artifactChanged = $currentArtifact -ne $ExpectedArtifact
$forceModelDownload = [bool]($Force -or $removedStaleSidecar -or $artifactChanged)

Download-File -RemotePath $ExpectedArtifact -LocalName 'model.onnx' -Required $true -ForceDownload $forceModelDownload

foreach ($file in @('tokenizer.json', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json')) {
    Download-File -RemotePath $file -LocalName $file -Required $true -ForceDownload ([bool]$Force)
}

foreach ($file in @('added_tokens.json', 'generation_config.json')) {
    Download-File -RemotePath $file -LocalName $file -Required $false -ForceDownload ([bool]$Force)
}

$manifestPath = Join-Path $TargetDir 'model-source.properties'
@"
repository=$Repo
revision=main
artifact=$ExpectedArtifact
localModelFile=model.onnx
externalDataFile=
format=q4f16-single-file-onnx
"@ | Set-Content -Encoding UTF8 -LiteralPath $manifestPath

if ($Validate) {
    Write-Host ""
    Write-Host "  File validation:"
    $requiredFiles = @('model.onnx', 'tokenizer.json', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json')
    $missing = @()
    foreach ($file in $requiredFiles) {
        $path = Join-Path $TargetDir $file
        if (Test-Path $path) {
            $size = (Get-Item -LiteralPath $path).Length
            $sizeStr = if ($size -gt 1MB) { "{0:N1} MB" -f ($size / 1MB) }
                       elseif ($size -gt 1KB) { "{0:N1} KB" -f ($size / 1KB) }
                       else { "$size B" }
            $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.Substring(0, 12)
            Write-Host ("    {0,-30} {1,10}  sha256:{2}" -f $file, $sizeStr, $hash)
        } else {
            $missing += $file
        }
    }
    if ($missing.Count -gt 0) {
        throw "Required files missing after download: $($missing -join ', ')"
    }
}

Write-Host ""
Write-Host "Qwen2.5-Coder 0.5B q4f16 model ready at: $TargetDir"
Write-Host "Expected loader log marker: Model format: INT4 quantized (MatMulNBits)"
