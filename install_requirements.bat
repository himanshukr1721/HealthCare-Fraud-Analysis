@echo off
echo ========================================
echo Healthcare Fraud Detection - Installation
echo ========================================
echo.

echo Step 1: Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)
echo.

echo Step 2: Installing core dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some packages from requirements.txt may have failed
    echo Continuing with PyTorch installation...
)
echo.

echo Step 3: Checking PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__)" 2>nul
if errorlevel 1 (
    echo PyTorch not found. Installing PyTorch (CPU version)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo ERROR: Failed to install PyTorch
        pause
        exit /b 1
    )
) else (
    echo PyTorch is already installed.
)
echo.

echo Step 4: Installing PyTorch Geometric...
pip install torch-geometric
if errorlevel 1 (
    echo ERROR: Failed to install torch-geometric
    echo You may need to install it manually
)
echo.

echo Step 5: Installing PyTorch Geometric extensions...
echo This may take a few minutes...
python -c "import torch; v=torch.__version__.split('+')[0]; print(f'Installing for PyTorch {v}')" 2>nul

REM Try to install extensions - they may fail if versions don't match exactly
pip install torch-scatter torch-sparse torch-cluster --no-index --find-links https://data.pyg.org/whl/torch-2.0.0+cpu.html 2>nul
if errorlevel 1 (
    echo Attempting alternative installation method...
    pip install torch-scatter torch-sparse torch-cluster
)
echo.

echo Step 6: Verifying installation...
python -c "import torch; import torch_geometric; import pandas; import streamlit; print('SUCCESS: Core packages installed!')" 2>nul
if errorlevel 1 (
    echo WARNING: Some packages may not be installed correctly
    echo Check the error messages above
) else (
    echo.
    echo ========================================
    echo Installation completed successfully!
    echo ========================================
)
echo.
pause
