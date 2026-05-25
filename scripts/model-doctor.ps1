# Model-Doctor: validates that a downloaded model directory is complete and
# immediately usable by the sidecar runtime.
#
# Usage:
#   pwsh scripts/model-doctor.ps1 -ModelDir model/all-MiniLM-L6-v2
#   pwsh scripts/model-doctor.ps1 -ModelDir model/e5-base-sts-en-de
#   pwsh scripts/model-doctor.ps1 -ModelDir model/cross-encoder-ms-marco-MiniLM-L-6-v2
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ModelDir
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ModelDir)) {
    Write-Error "Model directory does not exist: $ModelDir"
    exit 1
}

$ModelDir = (Resolve-Path -LiteralPath $ModelDir).Path

Write-Host "Model-Doctor: checking $ModelDir"
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
            $sizeStr = if ($size -gt 1MB) { "{0:N1} MB" -f ($size / 1MB) }
                       elseif ($size -gt 1KB) { "{0:N1} KB" -f ($size / 1KB) }
                       else { "$size B" }
            Write-Host "  [OK] $f ($sizeStr)"
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
        $sizeStr = if ($size -gt 1MB) { "{0:N1} MB" -f ($size / 1MB) }
                   elseif ($size -gt 1KB) { "{0:N1} KB" -f ($size / 1KB) }
                   else { "$size B" }
        Write-Host "  [OK] $f ($sizeStr)"
    }
}

# --- Validate config.json is valid JSON ---
$configPath = Join-Path $ModelDir 'config.json'
if (Test-Path $configPath) {
    try {
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        if ($config.hidden_size) {
            Write-Host "  [OK] config.json: hidden_size=$($config.hidden_size)"
        }
        if ($config.num_hidden_layers) {
            Write-Host "  [OK] config.json: num_hidden_layers=$($config.num_hidden_layers)"
        }
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
} else {
    Write-Host ""
    Write-Host "  Model-Doctor: ALL CHECKS PASSED. Model is ready." -ForegroundColor Green
    exit 0
}
