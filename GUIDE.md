# Developer & Maintainer Guide

## Healthcare Fraud Detection System

This guide provides comprehensive information for developers and maintainers working on the Healthcare Fraud Detection project. It covers architecture, code structure, extension points, and maintenance procedures.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Core Components](#core-components)
5. [Environment Setup](#environment-setup)
6. [Running the System](#running-the-system)
7. [Code Architecture Details](#code-architecture-details)
8. [Extension Points](#extension-points)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Project Overview

This project implements a **healthcare claims fraud detection pipeline** using:

- **Data Preprocessing**: Comprehensive cleaning and feature engineering
- **Knowledge Graph Construction**: Building heterogeneous graphs from healthcare data
- **Graph Neural Network (GNN)**: Deep learning model for fraud classification
- **Explainability**: SHAP, LIME, and GNN-specific explanations
- **Interactive Dashboard**: Streamlit-based web application

### Key Technologies

- **PyTorch Geometric**: Graph neural network framework
- **NetworkX**: Graph construction and analysis
- **Streamlit**: Interactive web interface
- **SHAP/LIME**: Model explainability tools

---

## Architecture

### High-Level Pipeline Flow

```
Raw Data (CSV)
    ↓
[Preprocessing] → Cleaned Data
    ↓
[Knowledge Graph Builder] → NetworkX Graph → PyTorch Geometric Data
    ↓
[GNN Model Training] → Trained Model
    ↓
[Explainability] → Explanations & Reports
```

### Component Interaction

```
main_pipeline.py (Orchestrator)
    ├── HealthcareDataPreprocessor (src/preprocessing.py)
    ├── HealthcareKnowledgeGraph (src/kg_gnn_model.py)
    ├── HealthcareGNN + GNNTrainer (src/kg_gnn_model.py)
    └── GNNExplainer (src/explainanbility.py)

streamlit_app.py (Interactive UI)
    └── Uses all above components via session state
```

---

## Project Structure

```
healthcare_kg_project/
├── data/
│   ├── raw/                          # Input data directory
│   │   └── healthcare_claims.csv     # Raw claims data
│   └── processed/                    # Processed data directory
│       └── cleaned_healthcare_data.csv
│
├── models/                           # Saved models directory
│   └── model_info.json               # Model metadata
│
├── outputs/                          # Generated outputs
│   ├── graphs/                       # Graph visualizations
│   ├── explanations/                 # Explanation reports
│   └── reports/                      # Analysis reports
│
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py              # Data preprocessing class
│   ├── kg_gnn_model.py               # KG builder, GNN model, trainer
│   ├── explainanbility.py           # Explainability tools
│   └── utils.py                      # Utility functions
│
├── venv/                             # Virtual environment (gitignored)
│
├── main_pipeline.py                  # Main batch pipeline
├── streamlit_app.py                  # Streamlit web application
├── requirements.txt                  # Python dependencies
├── requirements_base.txt             # Base dependencies
├── INSTALL.md                        # Installation instructions
├── README.md                         # User-facing documentation
└── GUIDE.md                          # This file
```

---

## Core Components

### 1. Main Pipeline (`main_pipeline.py`)

**Purpose**: Orchestrates the complete end-to-end batch processing pipeline.

**Key Functions**:
- `run_preprocessing()`: Executes data cleaning and preprocessing
- `build_knowledge_graph()`: Constructs knowledge graph from cleaned data
- `train_model()`: Trains the GNN model
- `generate_explanations()`: Creates explainability reports
- `main()`: Entry point that runs all steps sequentially

**Usage**:
```bash
python main_pipeline.py
```

**Outputs**:
- `data/processed/cleaned_healthcare_data.csv`
- `models/model_info.json`
- `outputs/explanations/explanations.json` + visual reports

---

### 2. Streamlit Application (`streamlit_app.py`)

**Purpose**: Interactive web-based interface for data exploration and model interaction.

**Modules** (via sidebar navigation):
1. **📤 Data Upload**: Upload and preview CSV files
2. **🔧 Preprocessing**: Run preprocessing pipeline interactively
3. **🕸️ Knowledge Graph**: Visualize and explore the knowledge graph
4. **🤖 GNN Analysis**: Train model and view predictions
5. **📊 Insights & Explainability**: SHAP/LIME explanations and visualizations

**Session State Variables**:
- `data_loaded`: Boolean flag for data availability
- `df_cleaned`: Cleaned DataFrame
- `kg`: Knowledge graph object (NetworkX)
- `model`: Trained GNN model
- `predictions`: Model predictions

**Usage**:
```bash
streamlit run streamlit_app.py
```

---

### 3. Data Preprocessing (`src/preprocessing.py`)

**Class**: `HealthcareDataPreprocessor`

**Responsibilities**:
- Load CSV data from file path
- Clean column names (strip whitespace)
- Handle missing values (median for numeric, mode for categorical)
- Remove duplicate claims (by `ClaimID`)
- Validate dates (`ClaimDate`, `DiagnosisDate`)
- Validate claim amounts (remove negatives, flag outliers)
- Standardize categorical variables (gender, status, type)
- Create derived features (age groups, claim categories, time features)
- Detect anomalies and flag potential fraud
- Generate summary statistics

**Key Methods**:
```python
preprocessor = HealthcareDataPreprocessor(filepath)
cleaned_data = preprocessor.preprocess()  # Full pipeline
preprocessor.save_cleaned_data(output_path)  # Save results
```

**Expected Input Schema**:
- Required: `ClaimID`, `PatientID`, `ProviderID`, `ClaimDate`, `ClaimAmount`
- Optional: `DiagnosisCode`, `ProcedureCode`, `PatientAge`, `PatientGender`, `ClaimStatus`, `ClaimType`, `ProviderSpecialty`

**Fraud Detection Heuristics**:
- Multiple claims by same patient on same day (>2)
- Claim amounts above 99th percentile

---

### 4. Knowledge Graph Builder (`src/kg_gnn_model.py`)

**Class**: `HealthcareKnowledgeGraph`

**Responsibilities**:
- Build heterogeneous NetworkX graph from claims data
- Create nodes: patients, providers, diagnoses, procedures
- Create edges: patient-provider, patient-diagnosis, patient-procedure, provider-diagnosis, diagnosis-procedure
- Compute graph statistics (density, clustering, connected components)
- Convert to PyTorch Geometric format for GNN training

**Key Methods**:
```python
kg_builder = HealthcareKnowledgeGraph(df)
G = kg_builder.build_graph()  # NetworkX graph
graph_data = kg_builder.to_pytorch_geometric(df)  # PyG Data object
```

**Graph Structure**:
- **Nodes**: 
  - `patient_{PatientID}`: Patient entities
  - `provider_{ProviderID}`: Healthcare provider entities
  - `diagnosis_{DiagnosisCode}`: Diagnosis code entities
  - `procedure_{ProcedureCode}`: Procedure code entities
- **Edges**:
  - `visited`: Patient → Provider (with claim_amount, claim_status)
  - `has_diagnosis`: Patient → Diagnosis
  - `received_procedure`: Patient → Procedure
  - `treats`: Provider → Diagnosis
  - `requires`: Diagnosis → Procedure

**PyTorch Geometric Conversion**:
- Creates `Data` object with:
  - `x`: Node feature matrix (encoded features)
  - `edge_index`: Edge connectivity (COO format)
  - `y`: Node labels (fraud/legitimate)
  - Train/validation/test splits

---

### 5. GNN Model (`src/kg_gnn_model.py`)

**Classes**: `HealthcareGNN`, `GNNTrainer`

**HealthcareGNN Architecture**:
- Input layer: Node features from knowledge graph
- Hidden layers: Multiple GCN/SAGE/GAT layers with batch normalization
- Output layer: Binary classification (fraud/legitimate)
- Activation: ReLU with dropout regularization

**Default Hyperparameters** (in `main_pipeline.py`):
- `input_dim`: Auto-detected from feature matrix
- `hidden_dim`: 64
- `output_dim`: 2 (binary classification)
- `num_layers`: 3
- `dropout`: 0.3

**GNNTrainer Responsibilities**:
- Train/validation split
- Training loop with optimization
- Evaluation and metrics (accuracy)
- Device management (CPU/GPU)
- Model checkpointing

**Usage**:
```python
model = HealthcareGNN(input_dim, hidden_dim=64, output_dim=2, num_layers=3, dropout=0.3)
trainer = GNNTrainer(model, graph_data)
final_acc = trainer.train(epochs=200)
```

---

### 6. Explainability (`src/explainanbility.py`)

**Class**: `GNNExplainer`

**Responsibilities**:
- Generate explanations for GNN predictions
- Extract node embeddings
- Compute feature importance (gradient-based)
- Generate explanation reports with visualizations
- Integrate SHAP and LIME for tabular feature explanations

**Key Methods**:
```python
explainer = GNNExplainer(model, graph_data, feature_names)
report = explainer.generate_explanation_report(node_idx, output_dir)
```

**Outputs**:
- Feature importance plots
- Node embedding visualizations
- Explanation JSON files
- SHAP/LIME plots (if applicable)

**Note**: Feature names must match the order and content of `graph_data.x` feature matrix.

---

### 7. Utilities (`src/utils.py`)

**Purpose**: Centralized utility functions for path management, file I/O, and helper operations.

**Key Functions**:

**Path Management**:
- `get_project_root()`: Returns project root directory
- `get_data_path(filename, subfolder='raw')`: Path to data files
- `get_output_path(filename, subfolder='reports')`: Path to output files
- `get_model_path(filename)`: Path to model files

**File I/O**:
- `save_json(data, filepath)`: Save dictionary to JSON
- `load_json(filepath)`: Load JSON file
- `save_pickle(obj, filepath)`: Save Python object via pickle
- `load_pickle(filepath)`: Load pickled object
- `ensure_dir(path)`: Create directory if it doesn't exist

**Helper Functions**:
- `print_separator(char='=', length=60)`: Print separator line
- `format_currency(amount)`: Format number as currency string
- `calculate_statistics(df, column)`: Compute column statistics

**Important**: Always use these utility functions for path construction to maintain consistency across the codebase.

---

## Environment Setup

### Python Version

**Recommended**: Python 3.8+ (project uses Python 3.13 based on venv)

### Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

### Dependencies Installation

**Option 1: Standard Installation**
```bash
pip install -r requirements.txt
```

**Option 2: Use Helper Scripts**
```bash
# Windows
install_requirements.bat

# Linux/macOS
bash install_requirements.sh
```

**Option 3: Manual PyTorch Geometric Setup**

PyTorch Geometric requires special installation:

```bash
# Install PyTorch first (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric

# Install PyG dependencies
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

For GPU (CUDA) support, replace `cpu` with `cu118` in the PyTorch installation URL.

### Verify Installation

Run the installation checker:
```bash
python check_installation.py
```

---

## Running the System

### Batch Pipeline Execution

**Full Pipeline**:
```bash
python main_pipeline.py
```

This executes:
1. Data preprocessing → `data/processed/cleaned_healthcare_data.csv`
2. Knowledge graph construction → NetworkX graph + PyG Data
3. GNN model training → Trained model + `models/model_info.json`
4. Explanation generation → `outputs/explanations/`

**Individual Steps** (programmatic):
```python
from main_pipeline import run_preprocessing, build_knowledge_graph, train_model

# Step 1: Preprocessing
cleaned_data = run_preprocessing()

# Step 2: Build KG
kg_builder, graph_data = build_knowledge_graph(cleaned_data)

# Step 3: Train
model, graph_data, trainer = train_model(graph_data, cleaned_data)
```

### Streamlit Application

**Launch**:
```bash
streamlit run streamlit_app.py
```

**Access**: Open browser to `http://localhost:8501`

**Workflow**:
1. Upload CSV file in "📤 Data Upload" tab
2. Run preprocessing in "🔧 Preprocessing" tab
3. Build and visualize KG in "🕸️ Knowledge Graph" tab
4. Train model in "🤖 GNN Analysis" tab
5. View explanations in "📊 Insights & Explainability" tab

---

## Code Architecture Details

### Data Flow

1. **Raw Data** → `HealthcareDataPreprocessor.preprocess()` → **Cleaned DataFrame**
2. **Cleaned DataFrame** → `HealthcareKnowledgeGraph.build_graph()` → **NetworkX Graph**
3. **NetworkX Graph** → `HealthcareKnowledgeGraph.to_pytorch_geometric()` → **PyG Data**
4. **PyG Data** → `GNNTrainer.train()` → **Trained Model**
5. **Trained Model + PyG Data** → `GNNExplainer.generate_explanation_report()` → **Explanations**

### Key Design Decisions

#### 1. Path Management
- **Decision**: Centralize all path construction in `src/utils.py`
- **Rationale**: Ensures consistency, easy to update if directory structure changes
- **Usage**: Always use `get_data_path()`, `get_output_path()`, `get_model_path()`

#### 2. Data Contracts
- **Decision**: Assume specific column names in preprocessing and KG building
- **Required Columns**: `ClaimID`, `PatientID`, `ProviderID`, `ClaimDate`, `ClaimAmount`
- **If Schema Changes**: Update `HealthcareDataPreprocessor` and `HealthcareKnowledgeGraph` accordingly

#### 3. Model Architecture
- **Decision**: Configurable GNN with default hyperparameters in `main_pipeline.py`
- **Rationale**: Allows experimentation while maintaining defaults
- **To Modify**: Edit `HealthcareGNN` class or update `train_model()` defaults

#### 4. Explainability
- **Decision**: Hard-coded feature names list in `main_pipeline.generate_explanations()`
- **Note**: Must match `graph_data.x` feature matrix order
- **To Update**: Modify feature list when adding/removing features

### Error Handling

- **Pipeline Level**: `main_pipeline.main()` wraps execution in try/except with traceback
- **Component Level**: Individual components print warnings but may not halt execution
- **Streamlit**: Uses `st.error()` for user-facing errors

---

## Extension Points

### 1. Adding New Fraud Detection Heuristics

**Location**: `src/preprocessing.py` → `HealthcareDataPreprocessor.detect_anomalies()`

**Example**:
```python
def detect_anomalies(self):
    # Existing code...
    
    # NEW: Flag providers with unusually high claim rates
    if 'ProviderID' in self.df.columns:
        provider_claims = self.df.groupby('ProviderID').size()
        high_frequency = provider_claims[provider_claims > provider_claims.quantile(0.95)]
        mask = self.df['ProviderID'].isin(high_frequency.index)
        self.df.loc[mask, 'PotentialFraud'] = True
```

**Integration**: The `PotentialFraud` flag can be used as a feature in the GNN model.

---

### 2. Adding New Node/Edge Types to Knowledge Graph

**Location**: `src/kg_gnn_model.py` → `HealthcareKnowledgeGraph`

**Example - Adding Facility Nodes**:
```python
def create_nodes(self):
    # Existing patient, provider, diagnosis, procedure nodes...
    
    # NEW: Facility nodes
    offset = len(patients) + len(providers) + len(diagnoses) + len(procedures)
    facilities = self.df['FacilityID'].unique()
    for idx, facility in enumerate(facilities):
        self.G.add_node(f"facility_{facility}", 
                      node_type='facility',
                      node_id=facility)
        self.node_mappings[f"facility_{facility}"] = offset + idx

def create_edges(self):
    # Existing edges...
    
    # NEW: Provider-Facility relationship
    for _, row in self.df.iterrows():
        provider = f"provider_{row['ProviderID']}"
        facility = f"facility_{row['FacilityID']}"
        self.G.add_edge(provider, facility,
                      edge_type='works_at',
                      frequency=1)
```

**Update PyG Conversion**: Modify `to_pytorch_geometric()` to include new node types in feature matrix.

---

### 3. Modifying GNN Architecture

**Location**: `src/kg_gnn_model.py` → `HealthcareGNN`

**Example - Switching to GAT (Graph Attention Network)**:
```python
from torch_geometric.nn import GATConv

class HealthcareGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=8, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=dropout))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim * 8, output_dim, heads=1))
        
        self.dropout = dropout
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim * 8) for _ in range(num_layers - 1)])
```

**Update Training**: No changes needed if interface remains the same.

---

### 4. Adding New Output Formats

**Location**: Any component that generates outputs

**Example - Adding CSV Export for Explanations**:
```python
# In src/explainanbility.py or main_pipeline.py
import pandas as pd

def save_explanations_csv(explanations, output_path):
    """Save explanations as CSV"""
    df = pd.DataFrame(explanations)
    df.to_csv(output_path, index=False)
    print(f"Explanations saved to {output_path}")
```

**Use Utility**: Leverage `get_output_path()` for consistent path management.

---

### 5. Integrating External Systems

**Example - FastAPI Service**:
```python
# api_service.py (new file)
from fastapi import FastAPI, File, UploadFile
from src.preprocessing import HealthcareDataPreprocessor
from src.kg_gnn_model import HealthcareKnowledgeGraph, HealthcareGNN
import torch

app = FastAPI()

# Load saved model
model = HealthcareGNN(...)
model.load_state_dict(torch.load('models/trained_model.pth'))
model.eval()

@app.post("/predict")
async def predict_fraud(file: UploadFile):
    # Preprocess
    preprocessor = HealthcareDataPreprocessor(file.filename)
    cleaned_data = preprocessor.preprocess()
    
    # Build KG
    kg_builder = HealthcareKnowledgeGraph(cleaned_data)
    graph_data = kg_builder.to_pytorch_geometric(cleaned_data)
    
    # Predict
    with torch.no_grad():
        predictions = model(graph_data.x, graph_data.edge_index)
    
    return {"predictions": predictions.tolist()}
```

---

## Troubleshooting

### Common Issues

#### 1. **Data Schema Errors**

**Symptom**: `KeyError` for columns like `DiagnosisCode`, `ProviderID`, etc.

**Solution**:
1. Check input CSV has required columns
2. Verify column names match expected schema (case-sensitive)
3. Update `HealthcareDataPreprocessor` if using different column names
4. Update `HealthcareKnowledgeGraph` node/edge creation accordingly

**Prevention**: Add schema validation in `HealthcareDataPreprocessor.load_data()`

---

#### 2. **PyTorch Geometric Shape Mismatches**

**Symptom**: `RuntimeError: Expected tensor to have size X but got size Y`

**Solution**:
1. Verify `graph_data.x.shape[1]` matches `input_dim` in model initialization
2. Check `edge_index` is properly formatted (2 x num_edges)
3. Ensure node features and edge indices are on same device (CPU/GPU)
4. Review `to_pytorch_geometric()` conversion logic

**Debug**:
```python
print(f"Node features shape: {graph_data.x.shape}")
print(f"Edge index shape: {graph_data.edge_index.shape}")
print(f"Model input_dim: {model.input_dim}")
```

---

#### 3. **Device Mismatch Errors**

**Symptom**: `RuntimeError: Expected all tensors to be on the same device`

**Solution**:
1. Ensure model and data are on same device
2. In `GNNTrainer`, move model and data to device consistently
3. Check `GNNExplainer` uses same device as model

**Fix**:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
graph_data = graph_data.to(device)
```

---

#### 4. **Explainability Feature Mismatch**

**Symptom**: `ValueError: feature_names length doesn't match feature matrix`

**Solution**:
1. Verify `feature_names` list length equals `graph_data.x.shape[1]`
2. Ensure feature order matches the order in `to_pytorch_geometric()`
3. Update feature list in `main_pipeline.generate_explanations()`

**Debug**:
```python
print(f"Feature names count: {len(feature_names)}")
print(f"Feature matrix columns: {graph_data.x.shape[1]}")
```

---

#### 5. **Streamlit Not Reflecting Changes**

**Symptom**: Code changes don't appear in running Streamlit app

**Solution**:
1. Stop Streamlit (Ctrl+C)
2. Restart: `streamlit run streamlit_app.py`
3. Clear browser cache if CSS/layout changes don't show
4. Use "Rerun" button in Streamlit UI for quick updates

---

#### 6. **Memory Issues with Large Graphs**

**Symptom**: `RuntimeError: CUDA out of memory` or system slowdown

**Solution**:
1. Reduce batch size in training (if applicable)
2. Use CPU instead of GPU for smaller datasets
3. Implement graph sampling (subgraph extraction)
4. Optimize node feature matrix (remove unnecessary features)

**Optimization**:
```python
# Sample subgraph for large datasets
from torch_geometric.utils import subgraph

def sample_subgraph(data, num_nodes=1000):
    node_indices = torch.randperm(data.x.size(0))[:num_nodes]
    edge_index, edge_attr = subgraph(node_indices, data.edge_index, data.edge_attr)
    return Data(x=data.x[node_indices], edge_index=edge_index, y=data.y[node_indices])
```

---

#### 7. **Missing Dependencies**

**Symptom**: `ModuleNotFoundError: No module named 'torch_geometric'`

**Solution**:
1. Verify virtual environment is activated
2. Install missing package: `pip install torch-geometric`
3. Check `requirements.txt` includes all dependencies
4. Run `python check_installation.py` to verify setup

---

## Best Practices

### Code Organization

1. **Use Utility Functions**: Always use `get_data_path()`, `get_output_path()`, etc. from `utils.py`
2. **Consistent Error Handling**: Wrap critical operations in try/except with informative messages
3. **Logging**: Use `print()` for progress (consider upgrading to `logging` module for production)
4. **Type Hints**: Add type hints to function signatures (gradual adoption)

### Data Management

1. **Schema Validation**: Validate input data schema early in preprocessing
2. **Data Versioning**: Consider versioning processed datasets if schema changes
3. **Backup**: Keep raw data backups before preprocessing
4. **Documentation**: Document expected input schema in docstrings

### Model Development

1. **Hyperparameter Tuning**: Use configuration files (JSON/YAML) for hyperparameters
2. **Model Versioning**: Save model metadata (hyperparameters, accuracy) with model weights
3. **Experiment Tracking**: Consider using MLflow or similar for experiment tracking
4. **Validation**: Always use train/validation/test splits

### Testing

1. **Unit Tests**: Write unit tests for utility functions
2. **Integration Tests**: Test full pipeline with sample data
3. **Data Validation**: Test preprocessing with edge cases (missing values, outliers)
4. **Model Validation**: Verify model predictions are reasonable

### Documentation

1. **Docstrings**: Add docstrings to all classes and functions
2. **Comments**: Comment complex logic and non-obvious decisions
3. **README Updates**: Keep README.md updated with new features
4. **Changelog**: Maintain CHANGELOG.md for version history

### Performance

1. **Profiling**: Profile slow operations (e.g., graph construction)
2. **Caching**: Cache expensive computations (e.g., graph statistics)
3. **Vectorization**: Use NumPy/Pandas vectorized operations
4. **GPU Utilization**: Ensure GPU is used when available for training

---

## Version History

- **v1.0**: Initial release with basic pipeline
- **v1.1**: Added Streamlit interface
- **v1.2**: Enhanced explainability features

---

## Contributing

When contributing to this project:

1. **Follow Code Style**: Maintain consistency with existing code
2. **Update Documentation**: Update this guide and README.md for new features
3. **Test Changes**: Verify changes work with existing pipeline
4. **Add Examples**: Include usage examples for new features

---

## Additional Resources

- **PyTorch Geometric Documentation**: https://pytorch-geometric.readthedocs.io/
- **NetworkX Documentation**: https://networkx.org/documentation/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **SHAP Documentation**: https://shap.readthedocs.io/
- **LIME Documentation**: https://github.com/marcotcr/lime

---

## Support

For questions or issues:
1. Check this guide and README.md
2. Review code comments and docstrings
3. Open an issue in the repository
4. Contact the development team

---

**Last Updated**: 2024
**Maintainer**: Development Team


