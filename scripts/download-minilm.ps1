# Downloads sentence-transformers/all-MiniLM-L6-v2 into model/all-MiniLM-L6-v2.
# After this, EmbeddingReferenceTest is auto-enabled and the sidecar's
# `embed` method returns real 384-dim vectors.
#
# Usage:
#   pwsh scripts/download-minilm.ps1
#   pwsh scripts/download-minilm.ps1 -ModelRoot model -Force -Validate
[CmdletBinding()]
param(
    [string]$ModelRoot = (Join-Path $PSScriptRoot '..\\model'),
    [string]$Variant = 'all-MiniLM-L6-v2',
    [switch]$Force,
    [switch]$Validate
)
$ErrorActionPreference = 'Stop'

# Load shared helper
. (Join-Path $PSScriptRoot '_Download-HfModel.ps1')

$repo = 'sentence-transformers/all-MiniLM-L6-v2'
$targetDir = Join-Path $ModelRoot $Variant

Write-Host "Downloading MiniLM model from $repo"

$result = Download-HfModel `
    -Repo $repo `
    -TargetDir $targetDir `
    -RequiredFiles @('model.safetensors', 'tokenizer.json', 'config.json') `
    -OptionalFiles @('vocab.txt', 'special_tokens_map.json', 'tokenizer_config.json') `
    -Force:$Force `
    -Validate:$Validate

Write-Host ""
Write-Host "MiniLM model ready at: $result"

# Run Model-Doctor
Write-Host ""
& (Join-Path $PSScriptRoot 'model-doctor.ps1') -ModelDir $result

