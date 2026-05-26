$ErrorActionPreference = 'Stop'

$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appJar = Join-Path $appDir 'directml-workbench-all.jar'

if (-not (Test-Path $appJar)) {
    Write-Error 'directml-workbench-all.jar was not found next to this launcher.'
    exit 1
}

$candidates = @()
if ($env:JAVA_HOME) { $candidates += (Join-Path $env:JAVA_HOME 'bin/java.exe') }
if ($env:JAVA_HOME_21_X64) { $candidates += (Join-Path $env:JAVA_HOME_21_X64 'bin/java.exe') }
$candidates += 'java'

$selectedJava = $null
$selectedVersion = $null

foreach ($candidate in $candidates) {
    try {
        $versionText = & $candidate -version 2>&1 | Select-String 'version' | Select-Object -First 1
        if (-not $versionText) { continue }
        $version = [regex]::Match($versionText.ToString(), 'version "([^"]+)"').Groups[1].Value
        if (-not $version) { continue }
        $majorText = $version.Split('.')[0]
        if ($majorText -eq '1') { $majorText = $version.Split('.')[1] }
        $major = [int]$majorText
        if ($major -ge 21) {
            $selectedJava = $candidate
            $selectedVersion = $version
            break
        }
    } catch {
        continue
    }
}

if (-not $selectedJava) {
    Write-Host 'DirectML Workbench requires Java 21 or newer.'
    Write-Host 'Set JAVA_HOME or JAVA_HOME_21_X64 to a Java 21 installation, or place Java 21 first on PATH.'
    Write-Host ''
    Write-Host 'Detected candidates were not usable as Java 21 runtimes.'
    exit 1
}

Write-Host "Starting DirectML Workbench with Java $selectedVersion"
& $selectedJava --enable-preview --enable-native-access=ALL-UNNAMED -jar $appJar @args
