<#
.SYNOPSIS
    Tags a new core release and triggers the GitHub Actions Maven Central workflow.
.PARAMETER Version
    The version to release, e.g. "0.4.0", "1.0.0".
.EXAMPLE
    .\release.ps1 0.4.0
.NOTES
    Requires the following GitHub Actions secrets to be configured on the repo
    aresstack/win-directml-java:
      CENTRAL_USERNAME      Sonatype Central Portal user-token
      CENTRAL_PASSWORD      Sonatype Central Portal password-token
      GPG_PRIVATE_KEY       ASCII-armored GPG private key
      GPG_PASSPHRASE        passphrase of the GPG key

    New releases publish these nine modules from root build.gradle
    `publishableModules` in one aggregated Sonatype deployment:
      directml-config              shared config/registry module
      directml-model-catalog       model catalog metadata
      directml-model-package       model package/load contracts
      directml-windows-bindings    Java 21 FFM runtime module
      directml-encoder             Java 21 encoder/reranker module
      directml-runtime             Java 21 public API module
      directml-inference           experimental generation / Whisper API
      audio-dsp                    Java-8 PCM speech preparation and VAD
      directml-whisper-audio       optional Whisper integration with AskAI DSP

    Legacy beta sidecar / Java-8 bridge / workbench artifacts remain in source
    but are not published by new releases.

    The release.yml workflow on the new tag push runs:
      ./gradlew -Pversion=$Version -x test :publishAggregationToCentralPortal
    which builds, signs and uploads one bundle to the Sonatype Central Portal.
    Confirm CI before tagging: the publishing workflow skips tests.
    On a publishing timeout, check the Sonatype deployment state before retrying.
    See the README's Releases / Maven Central section for exact recovery steps.
#>
param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidatePattern('^\d+\.\d+\.\d+')]
    [string]$Version
)

$ErrorActionPreference = 'Stop'
$tag = "v$Version"

if (-not (Test-Path 'build.gradle')) {
    Write-Error "build.gradle not found. Run this script from the repository root."
}

$existingTag = git tag -l $tag
if ($existingTag) {
    Write-Error "Tag $tag already exists. Choose a different version."
}

# --- Update version in README.md (Gradle/Maven dependency snippets) ---
Write-Host "[1/4] Updating README.md ..." -ForegroundColor Cyan
$readmeBytes = [System.IO.File]::ReadAllBytes("$PWD\README.md")
$readmeText  = [System.Text.Encoding]::UTF8.GetString($readmeBytes)
$readmeNew   = $readmeText
$readmeNew   = $readmeNew -replace '(<version>)[^<]+(</version>)', "`${1}$Version`${2}"
$readmeNew   = $readmeNew -replace "(implementation\s+'com\.aresstack:[^:]+:)[^']+'", "`${1}$Version'"
$readmeNew   = $readmeNew -replace "(implementation\s+`"com\.aresstack:[^:]+:)[^`"]+`"", "`${1}$Version`""
if ($readmeNew -ne $readmeText) {
    [System.IO.File]::WriteAllBytes("$PWD\README.md", [System.Text.Encoding]::UTF8.GetBytes($readmeNew))
    Write-Host "       README.md -> $Version" -ForegroundColor Green
} else {
    Write-Host "       README.md already at $Version" -ForegroundColor Yellow
}

# --- Commit ---
Write-Host "[2/4] Committing ..." -ForegroundColor Cyan
git add README.md
$diff = git diff --cached --name-only
if ($diff) {
    git commit -m "release $Version"
} else {
    Write-Host "       No changes - skipping commit." -ForegroundColor Yellow
}

# --- Tag ---
Write-Host "[3/4] Creating tag $tag ..." -ForegroundColor Cyan
git tag $tag

# --- Push ---
Write-Host "[4/4] Pushing to origin ..." -ForegroundColor Cyan
git push origin HEAD --tags

Write-Host ""
Write-Host "Done! Tag $tag pushed." -ForegroundColor Green
Write-Host "GitHub Actions workflow will now build, sign and publish core inference artifacts."
Write-Host "Monitor: https://github.com/aresstack/win-directml-java/actions" -ForegroundColor Yellow

