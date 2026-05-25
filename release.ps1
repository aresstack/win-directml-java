<#
.SYNOPSIS
    Releases a new core version of win-directml-java to Maven Central via GitHub Actions.
.PARAMETER Version
    The version to release, e.g. "0.2.0", "1.0.0".
.EXAMPLE
    .\release.ps1 0.2.0
.NOTES
    Requires the following GitHub Actions secrets to be configured on the repo
    aresstack/win-directml-java:
      CENTRAL_USERNAME      Sonatype Central Portal user-token
      CENTRAL_PASSWORD      Sonatype Central Portal password-token
      GPG_PRIVATE_KEY       ASCII-armored GPG private key
      GPG_PASSPHRASE        passphrase of the GPG key

    New releases publish only the core Java 21 inference artifacts listed in
    root build.gradle `publishableModules`:
      directml-config
      directml-windows-bindings
      directml-encoder
      directml-runtime

    Legacy beta sidecar / Java-8 bridge / workbench artifacts remain in source
    but are not published by new releases.

    The release.yml workflow on the tag push runs:
      ./gradlew -Pversion=$Version -x test publishAllPublicationsToCentralPortal
    which builds, signs and uploads all currently publishable modules to the
    Sonatype Central Portal.
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
Write-Host "GitHub Actions workflow will now build, sign and publish core Maven artifacts."
Write-Host "Monitor: https://github.com/aresstack/win-directml-java/actions" -ForegroundColor Yellow

