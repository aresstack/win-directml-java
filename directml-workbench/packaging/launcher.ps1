$ErrorActionPreference = 'Stop'

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Jar = Join-Path $AppDir 'directml-workbench-all.jar'

if (-not (Test-Path $Jar)) {
    Write-Error "directml-workbench-all.jar was not found next to launcher.ps1"
    exit 1
}

function Get-JavaMajorFromVersionObject($Version) {
    if ($null -eq $Version) { return $null }
    try {
        if ($Version.Major -eq 1 -and $Version.Minor -gt 0) {
            return [int]$Version.Minor
        }
        return [int]$Version.Major
    } catch {
        return $null
    }
}

function Get-JavaMajorFromVersionText($JavaPath) {
    try {
        $VersionOutput = & $JavaPath -version 2>&1
        $VersionLine = $VersionOutput | Select-String 'version' | Select-Object -First 1
        if (-not $VersionLine) { return $null }

        $Text = $VersionLine.ToString()
        if ($Text -notmatch '"([^"]+)"') { return $null }

        $Parts = $Matches[1].Split('.')
        if ($Parts[0] -eq '1' -and $Parts.Length -gt 1) {
            return [int]$Parts[1]
        }
        return [int]$Parts[0]
    } catch {
        return $null
    }
}

$Candidates = @()

if ($env:JAVA_HOME_21_X64) {
    $Path = Join-Path $env:JAVA_HOME_21_X64 'bin/java.exe'
    if (Test-Path $Path) {
        $Candidates += [pscustomobject]@{ Source = $Path; Major = (Get-JavaMajorFromVersionText $Path) }
    }
}

if ($env:JAVA_HOME) {
    $Path = Join-Path $env:JAVA_HOME 'bin/java.exe'
    if (Test-Path $Path) {
        $Candidates += [pscustomobject]@{ Source = $Path; Major = (Get-JavaMajorFromVersionText $Path) }
    }
}

# PowerShell exposes all java.exe entries on PATH here, including entries after
# an older Java 8 installation. Prefer this metadata because it is exactly what
# users see with: Get-Command java -All
Get-Command java -All -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.Source) {
        $Candidates += [pscustomobject]@{
            Source = $_.Source
            Major = (Get-JavaMajorFromVersionObject $_.Version)
        }
    }
}

$SelectedJava = $null
$SelectedMajor = $null
$Seen = @{}

foreach ($Candidate in $Candidates) {
    $Java = $Candidate.Source
    if ([string]::IsNullOrWhiteSpace($Java)) { continue }
    if ($Seen.ContainsKey($Java)) { continue }
    $Seen[$Java] = $true
    if (-not (Test-Path $Java)) { continue }

    $Major = $Candidate.Major
    if ($null -eq $Major) {
        $Major = Get-JavaMajorFromVersionText $Java
    }
    if ($null -eq $Major) { continue }

    Write-Host "Found Java $Major at: $Java"

    if ($Major -ge 21) {
        $SelectedJava = $Java
        $SelectedMajor = $Major
        break
    }
}

if (-not $SelectedJava) {
    Write-Error "Java 21 or newer was not found. Install Java 21 or add it to PATH."
    exit 1
}

Write-Host "Starting DirectML Workbench with Java $SelectedMajor from: $SelectedJava"
# Note: --add-modules=jdk.incubator.vector is intentionally NOT passed. The product
# classes load without the incubator module (the Vector-API is isolated behind a
# reflectively-loaded impl; CPU SIMD paths fall back to scalar when it is absent).
# Compile/test still use the module. To opt into the Vector-API SIMD speed-up at
# runtime, add --add-modules=jdk.incubator.vector here or via JAVA_TOOL_OPTIONS.
& $SelectedJava --enable-preview --enable-native-access=ALL-UNNAMED -jar $Jar @args
exit $LASTEXITCODE
