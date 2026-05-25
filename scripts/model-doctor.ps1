# Model-Doctor: validates that a downloaded model directory is complete and
# compatible with the current sidecar runtime.
#
# Usage:
#   pwsh scripts/model-doctor.ps1 -ModelDir model/all-MiniLM-L6-v2 -Family minilm
#   pwsh scripts/model-doctor.ps1 -ModelDir model/e5-base-v2 -Family e5 -Variant base-v2
#   pwsh scripts/model-doctor.ps1 -ModelDir model/cross-encoder-ms-marco-MiniLM-L-6-v2 -Family reranker
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ModelDir,

    [ValidateSet('auto', 'minilm', 'e5', 'reranker')]
    [string]$Family = 'auto',

    [string]$Variant = ''
)

$ErrorActionPreference = 'Stop'

function Format-FileSize([long]$Size) {
    if ($Size -gt 1MB) { return "{0:N1} MB" -f ($Size / 1MB) }
    if ($Size -gt 1KB) { return "{0:N1} KB" -f ($Size / 1KB) }
    return "$Size B"
}

function Get-ConfigValue([object]$Config, [string]$Name) {
    if ($null -eq $Config) { return $null }
    $prop = $Config.PSObject.Properties[$Name]
    if ($null -eq $prop) { return $null }
    return $prop.Value
}

function Expect-Int([object]$Config, [string]$Name, [int]$Expected, [ref]$Errors, [ref]$Warnings) {
    $value = Get-ConfigValue $Config $Name
    if ($null -eq $value) {
        $Warnings.Value += "config.json does not declare $Name; expected $Expected for this runtime family"
        return
    }
    if ([int]$value -ne $Expected) {
        $Errors.Value += "config.json mismatch: $Name=$value, expected $Expected"
    }
}

if (-not (Test-Path $ModelDir)) {
    Write-Error "Model directory does not exist: $ModelDir"
    exit 1
}

$ModelDir = (Resolve-Path -LiteralPath $ModelDir).Path
$leaf = Split-Path -Leaf $ModelDir

if ($Family -eq 'auto') {
    if ($leaf -like 'e5-*') { $Family = 'e5' }
    elseif ($leaf -like 'all-MiniLM*' -or $leaf -like '*MiniLM-L6-v2') { $Family = 'minilm' }
    elseif ($leaf -like 'cross-encoder-*' -or $leaf -like '*reranker*') { $Family = 'reranker' }
}

Write-Host "Model-Doctor: checking $ModelDir"
Write-Host "  family: $Family"
if (-not [string]::IsNullOrWhiteSpace($Variant)) {
    Write-Host "  variant: $Variant"
}
Write-Host ""

$errors = @()
$warnings = @()

# --- Required files ---
$required = @('config.json', 'tokenizer.json', 'model.safetensors')
foreach ($f in $required) {
    $path = Join-Path $ModelDir $f
    if (-not (Test-Path $path)) {
        $errors += "MISSING required file: $f"
    } else {
        $size = (Get-Item $path).Length
        if ($size -eq 0) {
            $errors += "EMPTY required file: $f (0 bytes)"
        } else {
            Write-Host "  [OK] $f ($(Format-FileSize $size))"
        }
    }
}

# --- Optional but recommended files ---
$optional = @('vocab.txt', 'special_tokens_map.json', 'tokenizer_config.json')
foreach ($f in $optional) {
    $path = Join-Path $ModelDir $f
    if (-not (Test-Path $path)) {
        $warnings += "optional file not present: $f"
    } else {
        $size = (Get-Item $path).Length
        Write-Host "  [OK] $f ($(Format-FileSize $size))"
    }
}

# --- Validate config.json is valid JSON and compatible ---
$config = $null
$configPath = Join-Path $ModelDir 'config.json'
if (Test-Path $configPath) {
    try {
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        $hidden = Get-ConfigValue $config 'hidden_size'
        $layers = Get-ConfigValue $config 'num_hidden_layers'
        $heads = Get-ConfigValue $config 'num_attention_heads'
        if ($hidden) { Write-Host "  [OK] config.json: hidden_size=$hidden" }
        if ($layers) { Write-Host "  [OK] config.json: num_hidden_layers=$layers" }
        if ($heads) { Write-Host "  [OK] config.json: num_attention_heads=$heads" }
    } catch {
        $errors += "config.json is not valid JSON: $($_.Exception.Message)"
    }
}

# --- Validate tokenizer.json is valid JSON ---
$tokenizerPath = Join-Path $ModelDir 'tokenizer.json'
if (Test-Path $tokenizerPath) {
    try {
        $null = Get-Content $tokenizerPath -Raw | ConvertFrom-Json
        Write-Host "  [OK] tokenizer.json: valid JSON"
    } catch {
        $errors += "tokenizer.json is not valid JSON: $($_.Exception.Message)"
    }
}

# --- Runtime-compatibility checks ---
$modelType = Get-ConfigValue $config 'model_type'
if ($modelType -and ($modelType -match 'xlm|roberta')) {
    $errors += "current runtime is WordPiece/BERT-style only for this path; config.json model_type=$modelType requires the planned SentencePiece/XLM-R path"
}

if ($Family -eq 'e5') {
    if ($Variant -eq 'base-sts-en-de') {
        $errors += "E5 variant base-sts-en-de is planned, not shipped: upstream is XLM-R/SentencePiece and must not be treated as a ready WordPiece runtime model"
    } elseif ($Variant -eq 'small-v2') {
        Expect-Int $config 'hidden_size' 384 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'num_hidden_layers' 12 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'num_attention_heads' 12 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'intermediate_size' 1536 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'type_vocab_size' 2 ([ref]$errors) ([ref]$warnings)
    } elseif ($Variant -eq 'base-v2' -or [string]::IsNullOrWhiteSpace($Variant)) {
        Expect-Int $config 'hidden_size' 768 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'num_hidden_layers' 12 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'num_attention_heads' 12 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'intermediate_size' 3072 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'type_vocab_size' 2 ([ref]$errors) ([ref]$warnings)
    } elseif ($Variant -eq 'large-v2') {
        Expect-Int $config 'hidden_size' 1024 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'num_hidden_layers' 24 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'num_attention_heads' 16 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'intermediate_size' 4096 ([ref]$errors) ([ref]$warnings)
        Expect-Int $config 'type_vocab_size' 2 ([ref]$errors) ([ref]$warnings)
    }
}

if ($Family -eq 'minilm') {
    Expect-Int $config 'hidden_size' 384 ([ref]$errors) ([ref]$warnings)
    Expect-Int $config 'num_hidden_layers' 6 ([ref]$errors) ([ref]$warnings)
    Expect-Int $config 'num_attention_heads' 12 ([ref]$errors) ([ref]$warnings)
    Expect-Int $config 'type_vocab_size' 2 ([ref]$errors) ([ref]$warnings)
}

if ($Family -eq 'reranker') {
    if ($modelType -and ($modelType -match 'xlm')) {
        $errors += "SentencePiece/XLM-R rerankers are not supported by the current WordPiece reranker path"
    }
}

# --- Report ---
Write-Host ""
if ($warnings.Count -gt 0) {
    foreach ($w in $warnings) {
        Write-Host "  [WARN] $w" -ForegroundColor Yellow
    }
}

if ($errors.Count -gt 0) {
    Write-Host ""
    foreach ($e in $errors) {
        Write-Host "  [FAIL] $e" -ForegroundColor Red
    }
    Write-Host ""
    Write-Error "Model-Doctor: $($errors.Count) error(s) found. Model is NOT ready."
    exit 1
}

Write-Host ""
Write-Host "  Model-Doctor: ALL CHECKS PASSED. Model is ready." -ForegroundColor Green
exit 0
