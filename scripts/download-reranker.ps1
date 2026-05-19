[CmdletBinding()]
param(
    [string]$Repo = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    [string]$Destination = 'model/cross-encoder-ms-marco-MiniLM-L-6-v2'
)

# Downloads a HuggingFace cross-encoder reranker checkpoint into
# `$Destination`. The sidecar discovers the model automatically when
# the directory contains `config.json`, `tokenizer.json` and
# `model.safetensors`.
#
# Typical models (BERT-WordPiece, drop-in compatible):
#   - cross-encoder/ms-marco-MiniLM-L-6-v2   (~90 MB, default)
#   - cross-encoder/ms-marco-MiniLM-L-12-v2  (~135 MB)
#   - BAAI/bge-reranker-base                  (~1.1 GB)
#
# SentencePiece-only families (e.g. bge-reranker-v2-m3) are NOT yet
# supported – the WordPiece tokenizer in directml-encoder rejects
# them at load time.

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Destination)) {
    New-Item -ItemType Directory -Path $Destination | Out-Null
}

$files = @('config.json', 'tokenizer.json', 'tokenizer_config.json',
           'special_tokens_map.json', 'vocab.txt', 'model.safetensors')

foreach ($f in $files) {
    $url = "https://huggingface.co/$Repo/resolve/main/$f"
    $out = Join-Path $Destination $f
    if (Test-Path $out) {
        Write-Host "skip $f (already present)"
        continue
    }
    Write-Host "fetching $f from $Repo..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
    } catch {
        # vocab.txt / tokenizer_config.json are optional on some repos.
        if ($f -in @('vocab.txt', 'tokenizer_config.json', 'special_tokens_map.json')) {
            Write-Warning "optional file $f not available, skipping"
            continue
        }
        throw
    }
}

Write-Host "Reranker model ready in: $Destination"

