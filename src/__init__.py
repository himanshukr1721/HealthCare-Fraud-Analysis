"""
Healthcare Fraud Detection Package
"""

# Import preprocessing (no heavy dependencies)
from .preprocessing import HealthcareDataPreprocessor

# Import KG and GNN models (requires torch_geometric)
try:
    from .kg_gnn_model import HealthcareKnowledgeGraph, HealthcareGNN, GNNTrainer
    _HAS_GNN = True
except ImportError:
    _HAS_GNN = False
    HealthcareKnowledgeGraph = None
    HealthcareGNN = None
    GNNTrainer = None

# Import explainability (requires torch, shap, lime)
try:
    from .explainanbility import GNNExplainer
    _HAS_EXPLAINER = True
except ImportError:
    _HAS_EXPLAINER = False
    GNNExplainer = None

__all__ = [
    'HealthcareDataPreprocessor',
    'HealthcareKnowledgeGraph',
    'HealthcareGNN',
    'GNNTrainer',
    'GNNExplainer',
    '_HAS_GNN',
    '_HAS_EXPLAINER'
]

