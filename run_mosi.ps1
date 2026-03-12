param(
    [switch]$Smoke,
    [switch]$SkipDataSetup
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $RepoRoot
try {
    $cmd = @("python", ".\scripts\run_mosi.py")
    if ($Smoke) {
        $cmd += "--smoke"
    }
    if ($SkipDataSetup) {
        $cmd += "--skip-data-setup"
    }
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
} finally {
    Pop-Location
}
