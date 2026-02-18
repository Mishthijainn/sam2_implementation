
$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'

$url = "https://github.com/facebookresearch/segment-anything-2/archive/refs/heads/main.zip"
$zipPath = "sam2.zip"
$extractPath = "segment-anything-2-main"
$finalPath = "segment-anything-2"

# 1. Download
if (Test-Path $zipPath) {
    if ((Get-Item $zipPath).Length -gt 50000000) {
        Write-Host "Zip file exists and seems valid. Skipping download."
    } else {
        Remove-Item $zipPath -Force
        Write-Host "Downloading $url..."
        Invoke-WebRequest -Uri $url -OutFile $zipPath
        Write-Host "Download complete."
    }
} else {
    Write-Host "Downloading $url..."
    Invoke-WebRequest -Uri $url -OutFile $zipPath
    Write-Host "Download complete."
}

if (Test-Path $finalPath) { Remove-Item $finalPath -Recurse -Force }
if (Test-Path $extractPath) { Remove-Item $extractPath -Recurse -Force }
if (Test-Path "sam2-main") { Remove-Item "sam2-main" -Recurse -Force }

# 2. Extract
Write-Host "Extracting..."
Expand-Archive -Path $zipPath -DestinationPath . -Force

# 3. Rename
if (Test-Path "sam2-main") {
    Rename-Item -Path "sam2-main" -NewName $finalPath
} elseif (Test-Path $extractPath) {
    Rename-Item -Path $extractPath -NewName $finalPath
} else {
    Write-Error "Could not find extracted folder!"
}

# 4. Install
Set-Location $finalPath
Write-Host "Installing SAM2..."
pip install -e .
pip install -e ".[dev]"
pip install requests matplotlib opencv-python jupyter
Set-Location ..

Write-Host "Done!"
