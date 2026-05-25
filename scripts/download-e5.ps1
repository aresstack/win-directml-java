# Downloads an E5 sentence-embedding checkpoint into model/<variant>.
# After this, E5RealModelReferenceTest is auto-enabled and the sidecar's
# `embed` method (with -Dembed.model=e5 -De5.model=<variant>) returns
# real vectors from the requested E5 family.
#
# Usage:
#   pwsh scripts/download-e5.ps1                          # default: base-sts-en-de
#   pwsh scripts/download-e5.ps1 -Variant small-v2
#   pwsh scripts/download-e5.ps1 -Variant base-v2
#   pwsh scripts/download-e5.ps1 -Variant large-v2
#   pwsh scripts/download-e5.ps1 -Variant base-sts-en-de -Force -Validate
#
# Variants are aligned with E5Variant.token() in the encoder runtime so
# the downloaded directory matches the directory hints used by
# resolveE5Dir() and E5Encoders.resolveConfig().
[CmdletBinding()]
param(
    [string]$ModelRoot = (Join-Path $PSScriptRoot '..\\model'),
    [ValidateSet('small-v2', 'base-v2', 'large-v2', 'base-sts-en-de')]
    [string]$Variant = 'base-sts-en-de',
    [switch]$Force,
    [switch]$Validate
)
$ErrorActionPreference = 'Stop'

# Load shared helper
. (Join-Path $PSScriptRoot '_Download-HfModel.ps1')

# Map a variant token to (huggingface-repo, local-folder).
$repoMap = @{
    'small-v2'       = @{ Repo = 'intfloat/e5-small-v2';            Folder = 'e5-small-v2' };
    'base-v2'        = @{ Repo = 'intfloat/e5-base-v2';             Folder = 'e5-base-v2' };
    'large-v2'       = @{ Repo = 'intfloat/e5-large-v2';            Folder = 'e5-large-v2' };
    'base-sts-en-de' = @{ Repo = 'danielheinz/e5-base-sts-en-de';   Folder = 'e5-base-sts-en-de' };
}
$entry = $repoMap[$Variant]
$targetDir = Join-Path $ModelRoot $entry.Folder

Write-Host "Downloading E5 variant '$Variant' from $($entry.Repo)"

$result = Download-HfModel `
    -Repo $entry.Repo `
    -TargetDir $targetDir `
    -RequiredFiles @('model.safetensors', 'tokenizer.json', 'config.json') `
    -OptionalFiles @('special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt') `
    -Force:$Force `
    -Validate:$Validate

Write-Host ""
Write-Host "E5 ($Variant) ready at: $result"
Write-Host "Use:  -Dembed.model=e5 -De5.model=$Variant -De5.modelDir=`"$result`""

# Run Model-Doctor
Write-Host ""
& (Join-Path $PSScriptRoot 'model-doctor.ps1') -ModelDir $result

