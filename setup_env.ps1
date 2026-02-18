$ErrorActionPreference = "Stop"

Write-Host "Starting SAM2 Setup..."

# 1. Install PyTorch (CPU version for stability first, or CUDA if available - defaulting to standard install which detects best)
# Using the stable URL for pytorch to avoid issues
Write-Host "Installing PyTorch..."
pip install torch torchvision torchaudio

# 2. Clone SAM2 Repository
if (-not (Test-Path "segment-anything-2")) {
    Write-Host "Cloning segment-anything-2 repository (shallow)..."
    git clone --depth 1 https://github.com/facebookresearch/segment-anything-2.git
} else {
    Write-Host "segment-anything-2 repository already exists."
}

# 3. Install SAM2
Write-Host "Installing SAM2..."
Set-Location segment-anything-2
# Install wrappers and other dependencies
pip install -e .
# Install dev dependencies just in case
pip install -e ".[dev]"
# Install visualization and notebook dependencies
pip install requests matplotlib opencv-python jupyter
Set-Location ..

Write-Host "SAM2 Setup Complete!"
