# Healthcare Fraud Detection using Knowledge Graph and Graph Neural Networks

A comprehensive system for detecting fraudulent healthcare insurance claims using Knowledge Graph construction and Graph Neural Network (GNN) models with explainability features.

## 🏗️ Project Structure

```
healthcare_kg_project/
├── data/
│   ├── raw/
│   │   └── healthcare_claims.csv
│   └── processed/
│       └── cleaned_healthcare_data.csv
├── models/
│   └── trained_gnn_model.pth
├── outputs/
│   ├── graphs/
│   ├── explanations/
│   └── reports/
├── src/
│   ├── preprocessing.py
│   ├── kg_gnn_model.py
│   ├── explainability.py
│   └── utils.py
├── streamlit_app.py
├── main_pipeline.py
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   
   # On Windows (Command Prompt)
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: PyTorch Geometric and related packages may require additional setup. If you encounter issues, install PyTorch first:
   ```bash
   pip install torch torchvision torchaudio
   ```

### Data Setup

Place your healthcare claims dataset in `data/raw/` as `healthcare_claims.csv`. The dataset should include columns such as:
- `ClaimID`, `PatientID`, `ProviderID`
- `ClaimDate`, `ClaimAmount`, `ClaimStatus`
- `PatientAge`, `PatientGender`
- `ProviderSpecialty`
- Other relevant features

## 📖 Usage

### 1. Run the Main Pipeline

Execute the complete fraud detection pipeline:

```bash
python main_pipeline.py
```

This will:
- Preprocess and clean the data
- Build a knowledge graph from the claims data
- Train a Graph Neural Network model
- Generate explanations for predictions

### 2. Run Preprocessing Only

To preprocess data separately:

```bash
python src/preprocessing.py
```

### 3. Launch Streamlit Web Application

Start the interactive web interface:

```bash
streamlit run streamlit_app.py
```

The app provides:
- Data overview and statistics
- Interactive data exploration
- Knowledge graph visualization
- Fraud detection results
- Model explanations

## 🔧 Key Components

### Preprocessing (`src/preprocessing.py`)
- Data cleaning and validation
- Missing value handling
- Feature engineering
- Anomaly detection
- Duplicate removal

### Knowledge Graph Builder (`src/kg_gnn_model.py`)
- `HealthcareKnowledgeGraph` class: Constructs knowledge graph from healthcare claims
- Creates relationships between patients, providers, diagnoses, and procedures
- Converts to PyTorch Geometric format for GNN training

### GNN Model (`src/kg_gnn_model.py`)
- `HealthcareGNN` class: Graph Convolutional Network (GCN) for fraud detection
- `GNNTrainer` class: Training and evaluation pipeline
- Trained on knowledge graph structure
- Predicts fraudulent claims using binary classification

### Explainability (`src/explainanbility.py`)
- SHAP (SHapley Additive exPlanations) integration
- LIME (Local Interpretable Model-agnostic Explanations)
- Graph-based explanations for node predictions
- GNNExplainer class for model interpretability

### Utilities (`src/utils.py`)
- Helper functions for file operations
- Path management
- Data loading and saving utilities

## 📊 Features

- **Knowledge Graph Construction**: Builds rich graph structures from healthcare data
- **Graph Neural Networks**: Leverages GCN/GAT for fraud detection
- **Explainability**: SHAP and LIME for model interpretability
- **Interactive Dashboard**: Streamlit-based web interface
- **Comprehensive Preprocessing**: Robust data cleaning pipeline

## 🔬 Model Architecture

The GNN model uses:
- **Input Layer**: Node features from knowledge graph
- **Hidden Layers**: Multiple GCN/GAT layers with batch normalization
- **Output Layer**: Binary classification (fraud/legitimate)
- **Activation**: ReLU with dropout for regularization

## 📈 Outputs

The pipeline generates:
- Cleaned dataset in `data/processed/`
- Trained model in `models/`
- Graph visualizations in `outputs/graphs/`
- Explanation reports in `outputs/explanations/`
- Analysis reports in `outputs/reports/`

## 🛠️ Configuration

### Environment Variables

Create a `.env` file for configuration (optional):
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Model Parameters

Modify model hyperparameters in `src/kg_gnn_model.py`:
- `hidden_dim`: Hidden layer dimension (default: 64)
- `num_layers`: Number of GNN layers (default: 3)
- `dropout`: Dropout rate (default: 0.5)

## 📝 Dependencies

Key dependencies include:
- **Core**: pandas, numpy, scikit-learn
- **Graph**: networkx, torch-geometric
- **Deep Learning**: PyTorch, torch-geometric
- **Explainability**: shap, lime
- **Visualization**: matplotlib, seaborn, plotly
- **Web App**: streamlit
- **Database**: neo4j (optional)

See `requirements.txt` for complete list with versions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- PyTorch Geometric team for GNN framework
- SHAP and LIME for explainability tools
- Streamlit for web app framework

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Note**: This is a research/educational project. For production use, ensure proper validation, testing, and compliance with healthcare data regulations.

