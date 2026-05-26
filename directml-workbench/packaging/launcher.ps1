$ErrorActionPreference = 'Stop'

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Jar = Join-Path $AppDir 'directml-workbench-all.jar'

if (-not (Test-Path $Jar)) {
    Write-Error "directml-workbench-all.jar was not found next to launcher.ps1"
    exit 1
}

$Candidates = @()
if ($env:JAVA_HOME_21_X64) { $Candidates += (Join-Path $env:JAVA_HOME_21_X64 'bin/java.exe') }
if ($env:JAVA_HOME) { $Candidates += (Join-Path $env:JAVA_HOME 'bin/java.exe') }
$PathJava = Get-Command java -ErrorAction SilentlyContinue
if ($PathJava) { $Candidates += $PathJava.Source }

foreach ($Java in $Candidates | Select-Object -Unique) {
    if ($Java -ne 'java' -and -not (Test-Path $Java)) { continue }

    $VersionOutput = & $Java -version 2>&1
    $VersionLine = $VersionOutput | Select-String 'version' | Select-Object -First 1
    if (-not $VersionLine) { continue }

    if ($VersionLine.ToString() -match '"([^".]*)\.?.*"') {
        $MajorText = $Matches[1]
        if ($MajorText -eq '1' -and $VersionLine.ToString() -match '"1\.([^".]*)') {
            $MajorText = $Matches[1]
        }
        $Major = [int]$MajorText
        if ($Major -ge 21) {
            Write-Host "Using Java $Major from: $Java"
            & $Java --enable-preview --enable-native-access=ALL-UNNAMED -jar $Jar @args
            exit $LASTEXITCODE
        }
    }
}

Write-Error "Java 21 or newer was not found. Set JAVA_HOME_21_X64 or JAVA_HOME to a Java 21 installation, or put Java 21 first on PATH."
exit 1
