# Downloads a HuggingFace cross-encoder reranker checkpoint.
# The sidecar discovers the model automatically when the directory
# contains `config.json`, `tokenizer.json` and `model.safetensors`.
#
# Typical models (BERT-WordPiece, drop-in compatible):
#   - cross-encoder/ms-marco-MiniLM-L-6-v2   (~90 MB, default)
#   - cross-encoder/ms-marco-MiniLM-L-12-v2  (~135 MB)
#   - BAAI/bge-reranker-base                  (~1.1 GB)
#
# SentencePiece-only families (e.g. bge-reranker-v2-m3) are NOT yet
# supported – the WordPiece tokenizer in directml-encoder rejects
# them at load time.
#
# Usage:
#   pwsh scripts/download-reranker.ps1
#   pwsh scripts/download-reranker.ps1 -Variant ms-marco-MiniLM-L-12-v2
#   pwsh scripts/download-reranker.ps1 -Force -Validate
[CmdletBinding()]
param(
    [string]$ModelRoot = (Join-Path $PSScriptRoot '..\\model'),
    [ValidateSet('ms-marco-MiniLM-L-6-v2', 'ms-marco-MiniLM-L-12-v2', 'bge-reranker-base')]
    [string]$Variant = 'ms-marco-MiniLM-L-6-v2',
    [switch]$Force,
    [switch]$Validate
)
$ErrorActionPreference = 'Stop'

# Load shared helper
. (Join-Path $PSScriptRoot '_Download-HfModel.ps1')

# Map variant to repo
$repoMap = @{
    'ms-marco-MiniLM-L-6-v2'  = @{ Repo = 'cross-encoder/ms-marco-MiniLM-L-6-v2';  Folder = 'cross-encoder-ms-marco-MiniLM-L-6-v2' };
    'ms-marco-MiniLM-L-12-v2' = @{ Repo = 'cross-encoder/ms-marco-MiniLM-L-12-v2'; Folder = 'cross-encoder-ms-marco-MiniLM-L-12-v2' };
    'bge-reranker-base'        = @{ Repo = 'BAAI/bge-reranker-base';                 Folder = 'bge-reranker-base' };
}
$entry = $repoMap[$Variant]
$targetDir = Join-Path $ModelRoot $entry.Folder

Write-Host "Downloading Reranker variant '$Variant' from $($entry.Repo)"

$result = Download-HfModel `
    -Repo $entry.Repo `
    -TargetDir $targetDir `
    -RequiredFiles @('model.safetensors', 'tokenizer.json', 'config.json') `
    -OptionalFiles @('tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt') `
    -Force:$Force `
    -Validate:$Validate

Write-Host ""
Write-Host "Reranker ($Variant) ready at: $result"

# Run Model-Doctor
Write-Host ""
& (Join-Path $PSScriptRoot 'model-doctor.ps1') -ModelDir $result

