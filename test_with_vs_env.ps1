# Test torch.compile with Visual Studio environment
# This script configures the VS environment and then runs the test

$vsPath = "C:\Program Files\Microsoft Visual Studio\18\Community"
$vcvarsPath = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"

if (Test-Path $vcvarsPath) {
    Write-Host "Setting up Visual Studio 2026 environment..." -ForegroundColor Green

    # Run vcvars64.bat and capture environment variables
    cmd /c "`"$vcvarsPath`" && set" | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            $name = $matches[1]
            $value = $matches[2]
            Set-Item -Path "env:$name" -Value $value
        }
    }

    Write-Host "Visual Studio environment configured!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Running torch.compile test..." -ForegroundColor Cyan
    Write-Host ""

    # Run the test with conda python
    & "C:\ProgramData\miniforge3\envs\python313\python.exe" test_torch_compile.py

} else {
    Write-Host "Error: vcvars64.bat not found at $vcvarsPath" -ForegroundColor Red
    exit 1
}
