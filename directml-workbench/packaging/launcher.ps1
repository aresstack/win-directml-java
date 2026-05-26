$ErrorActionPreference = 'Stop'

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Jar = Join-Path $AppDir 'directml-workbench-all.jar'

if (-not (Test-Path $Jar)) {
    Write-Error "directml-workbench-all.jar was not found next to launcher.ps1"
    exit 1
}

function Get-JavaMajorVersion($JavaPath) {
    try {
        $VersionOutput = & $JavaPath -version 2>&1
        $VersionLine = $VersionOutput | Select-String 'version' | Select-Object -First 1
        if (-not $VersionLine) { return $null }

        $VersionText = $VersionLine.ToString()
        if ($VersionText -notmatch '"([^"]+)"') { return $null }

        $Version = $Matches[1]
        $Parts = $Version.Split('.')
        if ($Parts[0] -eq '1' -and $Parts.Length -gt 1) {
            return [int]$Parts[1]
        }
        return [int]$Parts[0]
    } catch {
        return $null
    }
}

$Candidates = New-Object System.Collections.Generic.List[string]

if ($env:JAVA_HOME_21_X64) {
    $Candidates.Add((Join-Path $env:JAVA_HOME_21_X64 'bin/java.exe'))
}
if ($env:JAVA_HOME) {
    $Candidates.Add((Join-Path $env:JAVA_HOME 'bin/java.exe'))
}

# PowerShell can see every java.exe on PATH. This is important when Java 8 is
# first on PATH but Java 21 is also installed later, for example under Zulu.
Get-Command java -All -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.Source) {
        $Candidates.Add($_.Source)
    }
}

$SelectedJava = $null
$SelectedMajor = $null
$Seen = @{}

foreach ($Java in $Candidates) {
    if ([string]::IsNullOrWhiteSpace($Java)) { continue }
    if ($Seen.ContainsKey($Java)) { continue }
    $Seen[$Java] = $true

    if ($Java -ne 'java' -and -not (Test-Path $Java)) { continue }

    $Major = Get-JavaMajorVersion $Java
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
& $SelectedJava --enable-preview --enable-native-access=ALL-UNNAMED -jar $Jar @args
exit $LASTEXITCODE
