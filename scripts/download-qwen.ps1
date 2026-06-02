param(
    [string]$ModelDir = "$PSScriptRoot\..\models\qwen2.5-coder-0.5b-directml-int4",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$repo = "onnx-community/Qwen2.5-Coder-0.5B-Instruct"
$revision = "main"

function Join-HuggingFaceResolveUrl {
    param(
        [string]$Repository,
        [string]$Revision,
        [string]$RemotePath
    )

    return "https://huggingface.co/$Repository/resolve/$Revision/$RemotePath?download=true"
}

function Download-File {
    param(
        [string]$RemotePath,
        [string]$TargetPath,
        [switch]$ForceDownload
    )

    if ((Test-Path $TargetPath) -and -not $ForceDownload) {
        Write-Host "Keep existing file: $TargetPath"
        return
    }

    $targetDirectory = Split-Path -Parent $TargetPath
    if (-not (Test-Path $targetDirectory)) {
        New-Item -ItemType Directory -Path $targetDirectory | Out-Null
    }

    $temporaryPath = "$TargetPath.tmp"
    if (Test-Path $temporaryPath) {
        Remove-Item $temporaryPath -Force
    }

    $url = Join-HuggingFaceResolveUrl -Repository $repo -Revision $revision -RemotePath $RemotePath
    Write-Host "Download $RemotePath -> $TargetPath"
    Invoke-WebRequest -Uri $url -OutFile $temporaryPath
    Move-Item -Path $temporaryPath -Destination $TargetPath -Force
}

$resolvedModelDir = Resolve-Path -LiteralPath (New-Item -ItemType Directory -Force -Path $ModelDir)
$modelPath = Join-Path $resolvedModelDir "model.onnx"

# Remove stale dense external-data files. q4f16 is a single-file ONNX artifact.
$staleExternalDataFiles = @(
    (Join-Path $resolvedModelDir "model.onnx_data"),
    (Join-Path $resolvedModelDir "model.onnx.data")
)

foreach ($staleExternalDataFile in $staleExternalDataFiles) {
    if (Test-Path $staleExternalDataFile) {
        Write-Host "Remove stale dense sidecar: $staleExternalDataFile"
        Remove-Item $staleExternalDataFile -Force
    }
}

Download-File -RemotePath "onnx/model_q4f16.onnx" -TargetPath $modelPath -ForceDownload:$Force

# Keep tokenizer/config files next to the model so the existing launcher can use the same directory.
$metadataFiles = @(
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json"
)

foreach ($metadataFile in $metadataFiles) {
    $targetPath = Join-Path $resolvedModelDir $metadataFile
    Download-File -RemotePath $metadataFile -TargetPath $targetPath -ForceDownload:$Force
}

$manifestPath = Join-Path $resolvedModelDir "model-source.properties"
@"
repository=$repo
revision=$revision
artifact=onnx/model_q4f16.onnx
localModelFile=model.onnx
externalDataFile=
format=q4f16-single-file-onnx
"@ | Set-Content -Encoding UTF8 -Path $manifestPath

Write-Host ""
Write-Host "Qwen q4f16 model download completed."
Write-Host "Model directory: $resolvedModelDir"
Write-Host "Model file:      $modelPath"
Write-Host "External data:   not used for q4f16 single-file ONNX"
