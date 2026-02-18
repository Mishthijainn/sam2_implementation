$ErrorActionPreference = "Continue"

Write-Host "Starting SAM2 Experiments..."

# 1. Verify Installation
Write-Host "Verifying Environment..."
python verify_setup.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Verification failed! Please check installation."
    exit 1
}

# 2. Run Experiments
Write-Host "Running Experiments..."
python src/experiments.py

# 3. Check Results
if (Test-Path "results") {
    Write-Host "Experiments finished. Results in 'results/' directory."
    Get-ChildItem results -Recurse
} else {
    Write-Error "No results directory found!"
}
