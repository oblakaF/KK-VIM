# Builds the KK-VIM GUI as an installed Windows application and keeps it current (no Python commands for the user).
#
# Packaging approach of the project: the hatchling wheel of pyproject.toml with its [project.gui-scripts] entry point
# kkvim-gui = "kkvim_gui.app:main". Installing the wheel creates Scripts\kkvim-gui.exe, a windowed launcher (no
# console). The environment gets exactly the versions of uv.lock (extra "gui", no development groups).
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\build_gui.ps1 [-Target <folder>] [-Python 3.14] [-IfStale | -Check]
#
#   (no switch)  build and install now;
#   -IfStale     build only if the installation is missing or was built from other sources (used by KK-VIM-GUI.cmd);
#   -Check       build nothing: exit 0 if the installation is current, 3 if it is missing or stale;
#   -Fingerprint build nothing: print the source fingerprint (for tests) and exit 0.
#
# Result: <Target>\Scripts\kkvim-gui.exe (default Target: dist\kkvim-gui in the repository; dist\ is ignored by git) and
# <Target>\kkvim-gui-build.json with the source fingerprint. The fingerprint is the SHA256 over every file of src\
# (without __pycache__ and *.pyc), pyproject.toml, uv.lock, README.md, this script and the Python version: any change of
# the program or of its locked dependencies makes the installation stale, so an obsolete build is never started.
#
# Safe replacement: the new installation is built and checked in <Target>.staging (a relocatable environment), then
# <Target> is renamed to <Target>.previous, <Target>.staging to <Target>, the new program is started once (--help) from
# its final place, and only then <Target>.previous is removed. Any failure leaves the previous installation in <Target>
# (restored if it was already moved). Only <Target>, <Target>.staging, <Target>.previous and dist\build-gui are written.
# The script changes neither uv.lock nor the project environment .venv.
#
# Exit codes: 0 current or built; 1 build failed (previous installation kept); 2 uv not found; 3 (-Check) stale or missing.

param(
    [string]$Target = "",
    [string]$Python = "3.14",
    [switch]$IfStale,
    [switch]$Check,
    [switch]$Fingerprint
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Target) { $Target = Join-Path $root "dist\kkvim-gui" }
$Target = [System.IO.Path]::GetFullPath($Target)
$staging = "$Target.staging"
$previous = "$Target.previous"
$build = Join-Path $root "dist\build-gui"
$stampName = "kkvim-gui-build.json"
$exeRelative = "Scripts\kkvim-gui.exe"

function Invoke-Checked([string]$what, [scriptblock]$command) {
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

function Get-SourceFingerprint {
    $files = @()
    $files += Get-ChildItem -Path (Join-Path $root "src") -Recurse -File |
        Where-Object { $_.FullName -notmatch '\\__pycache__\\' -and $_.Extension -ne ".pyc" }
    foreach ($name in @("pyproject.toml", "uv.lock", "README.md")) {
        $path = Join-Path $root $name
        if (Test-Path -LiteralPath $path) { $files += Get-Item -LiteralPath $path }
    }
    $files += Get-Item -LiteralPath $PSCommandPath
    $lines = foreach ($file in $files) {
        $relative = $file.FullName.Substring($root.Length).TrimStart('\').Replace('\', '/')
        "{0}`t{1}" -f $relative, (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash
    }
    $sorted = [string[]]$lines
    [Array]::Sort($sorted, [StringComparer]::Ordinal)
    $text = "python=$Python`n" + ($sorted -join "`n")
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $bytes = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($text))
    return @{ fingerprint = ([BitConverter]::ToString($bytes) -replace "-", ""); files = $sorted.Count }
}

function Read-Stamp([string]$folder) {
    $path = Join-Path $folder $stampName
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    try { return Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json } catch { return $null }
}

function Test-Current([hashtable]$source) {
    if (-not (Test-Path -LiteralPath (Join-Path $Target $exeRelative))) { return "not installed" }
    $stamp = Read-Stamp $Target
    if ($null -eq $stamp) { return "installed without a build stamp (an earlier build): rebuild" }
    if ($stamp.fingerprint -ne $source.fingerprint) { return "built from other sources ($($stamp.built)): rebuild" }
    return $null
}

function Test-Program([string]$folder) {
    # the installed launcher must find its interpreter from where it is and import the program (exit code of --help)
    $exe = Join-Path $folder $exeRelative
    if (-not (Test-Path -LiteralPath $exe)) { throw "kkvim-gui.exe was not created: $exe" }
    $process = Start-Process -FilePath $exe -ArgumentList "--help" -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) { throw "kkvim-gui.exe --help failed in $folder (exit code $($process.ExitCode))" }
}

function Remove-Owned([string]$path) {
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
}

try {
    $source = Get-SourceFingerprint
    if ($Fingerprint) { Write-Output $source.fingerprint; exit 0 }
    $stale = Test-Current $source
    if ($Check) {
        if ($null -eq $stale) { Write-Output "KK-VIM GUI is current: $(Join-Path $Target $exeRelative)"; exit 0 }
        Write-Output "KK-VIM GUI: $stale"
        exit 3
    }
    if ($IfStale -and $null -eq $stale) {
        Write-Output "KK-VIM GUI is current: $(Join-Path $Target $exeRelative)"
        exit 0
    }
    if ($null -ne $stale) { Write-Output "KK-VIM GUI: $stale" }
    if ($null -eq (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Output "ERROR: uv was not found on PATH. Install uv (https://docs.astral.sh/uv/) and start again."
        if (Test-Path -LiteralPath (Join-Path $Target $exeRelative)) {
            Write-Output "The installed program in $Target is unchanged and was not started: it is not built from the current sources."
        }
        exit 2
    }
    Write-Output "Building KK-VIM GUI into $Target (versions of uv.lock) ..."
    Push-Location $root
    try {
        Invoke-Checked "uv lock --check" { uv lock --check }
        Remove-Owned $build
        New-Item -ItemType Directory -Force $build | Out-Null
        Invoke-Checked "uv build" { uv build --wheel --out-dir $build }
        $wheel = Get-ChildItem $build -Filter "kkvim-*.whl" | Select-Object -First 1
        if ($null -eq $wheel) { throw "no wheel in $build" }
        $requirements = Join-Path $build "requirements-gui.txt"
        Invoke-Checked "uv export" {
            uv export --locked --format requirements-txt --extra gui --no-dev --no-emit-project --no-hashes --output-file $requirements
        }
        Remove-Owned $staging
        Invoke-Checked "uv venv" { uv venv --relocatable --python $Python $staging }
        $stagingPython = Join-Path $staging "Scripts\python.exe"
        Invoke-Checked "install locked dependencies" { uv pip install --python $stagingPython -r $requirements }
        Invoke-Checked "install the kkvim wheel" { uv pip install --python $stagingPython --no-deps $wheel.FullName }
        $probe = "import sys, kkvim, kkvim_gui.app, PySide6, pyqtgraph, kkvim_navier_worker, kkvim_reference.navier; " +
                 "from kkvim_gui.backend.request import route_of; " +
                 "assert route_of('CCCC', 6) == 'cccc_n46' and route_of('SSSS', 4) == 'ssss_n46'; " +
                 "assert kkvim.__file__.startswith(sys.prefix) and kkvim_gui.app.__file__.startswith(sys.prefix); " +
                 "assert kkvim_navier_worker.__file__.startswith(sys.prefix); print('kkvim-gui ok')"
        Invoke-Checked "import check" { & $stagingPython -c $probe }
        Test-Program $staging
        $stamp = [ordered]@{
            fingerprint = $source.fingerprint; files = $source.files; python = $Python; built = (Get-Date).ToString("s")
            source = $root; wheel = $wheel.Name; uv = ((uv --version) -join " ")
        }
        $stamp | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $staging $stampName) -Encoding UTF8
    }
    catch {
        Remove-Owned $staging
        throw
    }
    finally {
        Pop-Location
    }
    # replace: the previous installation is kept until the new one has started from its final place
    Remove-Owned $previous
    $hadPrevious = Test-Path -LiteralPath $Target
    if ($hadPrevious) {
        try { Rename-Item -LiteralPath $Target -NewName (Split-Path $previous -Leaf) }
        catch {
            Remove-Owned $staging
            throw "the installed program in $Target is in use or locked; close KK-VIM and start again ($($_.Exception.Message))"
        }
    }
    try {
        Rename-Item -LiteralPath $staging -NewName (Split-Path $Target -Leaf)
        Test-Program $Target
    }
    catch {
        $message = $_.Exception.Message
        if (Test-Path -LiteralPath $Target) { Rename-Item -LiteralPath $Target -NewName (Split-Path $staging -Leaf) }
        if ($hadPrevious) { Rename-Item -LiteralPath $previous -NewName (Split-Path $Target -Leaf) }
        Remove-Owned $staging
        throw "the new installation did not start from $Target; the previous one is restored ($message)"
    }
    Remove-Owned $previous
    Write-Output "KK-VIM GUI: $(Join-Path $Target $exeRelative)"
    exit 0
}
catch {
    Write-Output "ERROR: KK-VIM GUI build failed: $($_.Exception.Message)"
    if (Test-Path -LiteralPath (Join-Path $Target $exeRelative)) {
        Write-Output "The previous installation in $Target is unchanged."
    }
    exit 1
}
