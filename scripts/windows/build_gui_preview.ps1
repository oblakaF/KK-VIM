# Builds the KK-VIM GUI functional preview as an installed Windows application (no Python commands for the user).
#
# Packaging approach of the project: the hatchling wheel of pyproject.toml with its [project.gui-scripts] entry point
# kkvim-gui = "kkvim_gui.app:main". Installing the wheel creates Scripts\kkvim-gui.exe, a windowed launcher (no
# console). The environment gets exactly the versions of uv.lock (extra "gui", no development groups).
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\build_gui_preview.ps1 [-Target <folder>] [-Python 3.14]
#
# Result: <Target>\Scripts\kkvim-gui.exe (default Target: dist\kkvim-gui-preview in the repository; dist\ is ignored
# by git). The script changes neither uv.lock nor the project environment .venv.

param(
    [string]$Target = "",
    [string]$Python = "3.14"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Target) { $Target = Join-Path $root "dist\kkvim-gui-preview" }
$build = Join-Path $root "dist\build-gui-preview"

function Invoke-Checked([string]$what, [scriptblock]$command) {
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

Push-Location $root
try {
    Invoke-Checked "uv lock --check" { uv lock --check }
    if (Test-Path $build) { Remove-Item -Recurse -Force $build }
    New-Item -ItemType Directory -Force $build | Out-Null
    Invoke-Checked "uv build" { uv build --wheel --out-dir $build }
    $wheel = Get-ChildItem $build -Filter "kkvim-*.whl" | Select-Object -First 1
    if ($null -eq $wheel) { throw "no wheel in $build" }
    $requirements = Join-Path $build "requirements-gui.txt"
    Invoke-Checked "uv export" {
        uv export --locked --format requirements-txt --extra gui --no-dev --no-emit-project --no-hashes --output-file $requirements
    }
    Invoke-Checked "uv venv" { uv venv --clear --python $Python $Target }
    $targetPython = Join-Path $Target "Scripts\python.exe"
    Invoke-Checked "install locked dependencies" { uv pip install --python $targetPython -r $requirements }
    Invoke-Checked "install the kkvim wheel" { uv pip install --python $targetPython --no-deps $wheel.FullName }
    $exe = Join-Path $Target "Scripts\kkvim-gui.exe"
    if (-not (Test-Path $exe)) { throw "kkvim-gui.exe was not created: $exe" }
    Invoke-Checked "import check" { & $targetPython -c "import kkvim, kkvim_gui.app, PySide6, pyqtgraph; print('kkvim-gui ok')" }
    Write-Output "KK-VIM GUI preview: $exe"
}
finally {
    Pop-Location
}
