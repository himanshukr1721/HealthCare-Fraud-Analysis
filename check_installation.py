#!/usr/bin/env python
"""Check if all required packages are installed"""

import sys

def check_package(name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = name
    try:
        __import__(import_name)
        print(f"[OK] {name}")
        return True
    except ImportError:
        print(f"[MISSING] {name}")
        return False

print("=" * 60)
print("Checking Healthcare Fraud Detection Dependencies")
print("=" * 60)
print()

all_ok = True

# Core dependencies
print("Core Dependencies:")
all_ok &= check_package("pandas")
all_ok &= check_package("numpy")
all_ok &= check_package("sklearn", "sklearn")
print()

# Graph libraries
print("Graph Libraries:")
all_ok &= check_package("networkx")
try:
    import community
    print("[OK] python-louvain")
except ImportError:
    print("[MISSING] python-louvain")
    all_ok = False
print()

# PyTorch
print("PyTorch:")
all_ok &= check_package("torch")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
except:
    pass

# PyTorch Geometric
print("PyTorch Geometric:")
all_ok &= check_package("torch_geometric")
try:
    import torch_geometric
    print(f"  PyG version: {torch_geometric.__version__}")
except:
    pass

all_ok &= check_package("torch_scatter")
all_ok &= check_package("torch_sparse")
all_ok &= check_package("torch_cluster")
print()

# Explainability
print("Explainability:")
all_ok &= check_package("shap")
all_ok &= check_package("lime")
print()

# Visualization
print("Visualization:")
all_ok &= check_package("matplotlib")
all_ok &= check_package("seaborn")
all_ok &= check_package("plotly")
print()

# Streamlit
print("Streamlit:")
all_ok &= check_package("streamlit")
print()

# Utilities
print("Utilities:")
all_ok &= check_package("tqdm")
try:
    import dotenv
    print("[OK] python-dotenv")
except ImportError:
    print("[MISSING] python-dotenv")
    all_ok = False
print()

print("=" * 60)
if all_ok:
    print("[SUCCESS] All packages are installed!")
    sys.exit(0)
else:
    print("[WARNING] Some packages are missing. Please install them.")
    print("\nTo install missing packages, run:")
    print("  install_requirements.bat  (Windows)")
    print("  ./install_requirements.sh  (Linux/Mac)")
    sys.exit(1)

