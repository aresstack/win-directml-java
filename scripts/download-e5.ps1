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
#   pwsh scripts/download-e5.ps1 -Variant base-sts-en-de
#
# Variants are aligned with E5Variant.token() in the encoder runtime so
# the downloaded directory matches the directory hints used by
# resolveE5Dir() and E5Encoders.resolveConfig().
param(
    [ValidateSet('small-v2', 'base-v2', 'large-v2', 'base-sts-en-de')]
    [string]$Variant = 'base-sts-en-de',
    [string]$Target
)
$ErrorActionPreference = 'Stop'

# Map a variant token to (huggingface-repo, local-folder).
$repoMap = @{
    'small-v2'       = @{ Repo = 'intfloat/e5-small-v2';            Folder = 'e5-small-v2' };
    'base-v2'        = @{ Repo = 'intfloat/e5-base-v2';             Folder = 'e5-base-v2' };
    'large-v2'       = @{ Repo = 'intfloat/e5-large-v2';            Folder = 'e5-large-v2' };
    'base-sts-en-de' = @{ Repo = 'danielheinz/e5-base-sts-en-de';   Folder = 'e5-base-sts-en-de' };
}
$entry = $repoMap[$Variant]
if (-not $Target -or [string]::IsNullOrWhiteSpace($Target)) {
    $Target = Join-Path $PSScriptRoot ('..\model\' + $entry.Folder)
}
$Target = (Resolve-Path -LiteralPath (New-Item -ItemType Directory -Force -Path $Target)).Path
$base = "https://huggingface.co/$($entry.Repo)/resolve/main"

# Files the encoder actually reads. config.json is *required* for E5:
# E5Encoders.resolveConfig() refuses to load without it so the variant
# choice can be cross-checked against the on-disk shapes.
$files = @('model.safetensors', 'tokenizer.json', 'config.json',
           'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt')

Write-Host "Downloading E5 variant '$Variant' from $($entry.Repo) into $Target"
foreach ($f in $files) {
    $dst = Join-Path $Target $f
    if (Test-Path $dst) { Write-Host "skip $f (exists)"; continue }
    try {
        Write-Host "downloading $f ..."
        Invoke-WebRequest -Uri "$base/$f" -OutFile $dst -UseBasicParsing
    } catch {
        # vocab.txt and special_tokens_map.json are optional on some
        # variants – only treat missing model.safetensors / tokenizer.json
        # / config.json as fatal.
        if ($f -in @('model.safetensors', 'tokenizer.json', 'config.json')) {
            throw "Failed to download required file '$f' from $base : $($_.Exception.Message)"
        }
        Write-Host "skip $f (not available on $($entry.Repo))"
    }
}
Write-Host "E5 ($Variant) ready at: $Target"
Write-Host "Use:  -Dembed.model=e5 -De5.model=$Variant -De5.modelDir=`"$Target`""

