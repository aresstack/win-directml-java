# Shared helper for downloading HuggingFace model files.
# Dot-source this from download-*.ps1 scripts.
#
# Provides: Download-HfModel function

function Download-HfModel {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Repo,

        [Parameter(Mandatory)]
        [string]$TargetDir,

        [Parameter(Mandatory)]
        [string[]]$RequiredFiles,

        [string[]]$OptionalFiles = @(),

        [switch]$Force,

        [switch]$Validate
    )

    $ErrorActionPreference = 'Stop'

    # Create target directory
    if (-not (Test-Path $TargetDir)) {
        New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    }
    $TargetDir = (Resolve-Path -LiteralPath $TargetDir).Path

    $base = "https://huggingface.co/$Repo/resolve/main"
    $allFiles = $RequiredFiles + $OptionalFiles

    foreach ($f in $allFiles) {
        $dst = Join-Path $TargetDir $f
        if ((Test-Path $dst) -and -not $Force) {
            Write-Host "  skip $f (exists, use -Force to re-download)"
            continue
        }
        try {
            Write-Host "  downloading $f ..."
            Invoke-WebRequest -Uri "$base/$f" -OutFile $dst -UseBasicParsing
        } catch {
            if ($f -in $RequiredFiles) {
                throw "Failed to download required file '$f' from $Repo : $($_.Exception.Message)"
            }
            Write-Host "  skip $f (not available on $Repo)"
        }
    }

    # Display file sizes if -Validate is set
    if ($Validate) {
        Write-Host ""
        Write-Host "  File validation:"
        $missing = @()
        foreach ($f in $allFiles) {
            $dst = Join-Path $TargetDir $f
            if (Test-Path $dst) {
                $size = (Get-Item $dst).Length
                $sizeStr = if ($size -gt 1MB) { "{0:N1} MB" -f ($size / 1MB) }
                           elseif ($size -gt 1KB) { "{0:N1} KB" -f ($size / 1KB) }
                           else { "$size B" }
                $hash = (Get-FileHash -Path $dst -Algorithm SHA256).Hash.Substring(0, 12)
                Write-Host ("    {0,-30} {1,10}  sha256:{2}" -f $f, $sizeStr, $hash)
            } elseif ($f -in $RequiredFiles) {
                $missing += $f
            }
        }
        if ($missing.Count -gt 0) {
            throw "Required files missing after download: $($missing -join ', ')"
        }
    }

    return $TargetDir
}
