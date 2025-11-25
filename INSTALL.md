# Installation Instructions

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation Steps

### Option 1: Automated Installation (Recommended)

**Windows:**
```bash
install_requirements.bat
```

**Linux/Mac:**
```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

### Option 2: Manual Installation

#### Step 1: Upgrade pip
```bash
python -m pip install --upgrade pip
```

#### Step 2: Install Core Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Install PyTorch
**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA (if you have NVIDIA GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Step 4: Install PyTorch Geometric and Dependencies

**Important:** PyTorch Geometric dependencies must be installed AFTER PyTorch.

**For CPU:**
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**For CUDA:**
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**Note:** Replace `torch-2.0.0` with your actual PyTorch version. Check with:
```bash
python -c "import torch; print(torch.__version__)"
```

#### Step 5: Install Optional Dependencies (if needed)

**Neo4j (only if using Neo4j database):**
```bash
pip install neo4j
```

## Troubleshooting

### Issue: `torch-scatter`, `torch-sparse`, or `torch-cluster` installation fails

**Solution:** These packages need to match your PyTorch version. 

1. Check your PyTorch version:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. Install matching versions from PyTorch Geometric's repository:
   ```bash
   pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-<YOUR_VERSION>+cpu.html
   ```

   Replace `<YOUR_VERSION>` with your PyTorch version (e.g., `2.0.0`, `2.1.0`)

### Issue: `ModuleNotFoundError: No module named 'torch_geometric'`

**Solution:** Make sure you installed PyTorch first, then PyTorch Geometric:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
```

### Issue: Version conflicts

**Solution:** Use a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

Then follow the installation steps above.

## Verify Installation

Test that all packages are installed correctly:

```bash
python -c "import torch; import torch_geometric; import pandas; import streamlit; print('All packages installed successfully!')"
```

## Quick Start

After installation, you can:

1. **Run the main pipeline:**
   ```bash
   python main_pipeline.py
   ```

2. **Launch Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

