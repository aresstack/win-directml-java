# Downloads sentence-transformers/all-MiniLM-L6-v2 into model/all-MiniLM-L6-v2.
# After this, EmbeddingReferenceTest is auto-enabled and the sidecar's
# `embed` method returns real 384-dim vectors.
#
# Usage:
#   pwsh scripts/download-minilm.ps1
param(
    [string]$Target = (Join-Path $PSScriptRoot '..\model\all-MiniLM-L6-v2')
)
$ErrorActionPreference = 'Stop'
$Target = (Resolve-Path -LiteralPath (New-Item -ItemType Directory -Force -Path $Target)).Path
$base = 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main'
$files = @('model.safetensors', 'tokenizer.json', 'config.json', 'vocab.txt',
           'special_tokens_map.json', 'tokenizer_config.json')
foreach ($f in $files) {
    $dst = Join-Path $Target $f
    if (Test-Path $dst) { Write-Host "skip $f (exists)"; continue }
    Write-Host "downloading $f ..."
    Invoke-WebRequest -Uri "$base/$f" -OutFile $dst -UseBasicParsing
}
Write-Host "MiniLM model ready at: $Target"

