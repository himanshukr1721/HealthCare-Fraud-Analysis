"""
Main pipeline for Healthcare Fraud Detection using Knowledge Graph and GNN
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.preprocessing import HealthcareDataPreprocessor
from src.kg_gnn_model import HealthcareKnowledgeGraph, HealthcareGNN, GNNTrainer
from src.explainanbility import GNNExplainer
from src.utils import (
    get_project_root, get_data_path, get_output_path, 
    get_model_path, save_json, print_separator
)
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


def run_preprocessing():
    """Step 1: Preprocess the raw data"""
    print_separator()
    print("STEP 1: DATA PREPROCESSING")
    print_separator()
    
    root = get_project_root()
    raw_data_path = get_data_path('healthcare_claims.csv', 'raw')
    processed_data_path = get_data_path('cleaned_healthcare_data.csv', 'processed')
    
    # Initialize preprocessor
    preprocessor = HealthcareDataPreprocessor(raw_data_path)
    
    # Run preprocessing
    cleaned_data = preprocessor.preprocess()
    
    # Save cleaned data
    preprocessor.save_cleaned_data(processed_data_path)
    
    return cleaned_data


def build_knowledge_graph(df: pd.DataFrame):
    """Step 2: Build knowledge graph from cleaned data"""
    print_separator()
    print("STEP 2: BUILDING KNOWLEDGE GRAPH")
    print_separator()
    
    kg_builder = HealthcareKnowledgeGraph(df)
    G = kg_builder.build_graph()
    
    # Convert to PyG data
    graph_data = kg_builder.to_pytorch_geometric(df)
    
    return kg_builder, graph_data


def train_model(graph_data: Data, df: pd.DataFrame):
    """Step 3: Train GNN model"""
    print_separator()
    print("STEP 3: TRAINING GNN MODEL")
    print_separator()
    
    # Initialize model
    input_dim = graph_data.x.shape[1]
    model = HealthcareGNN(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=2,  # Binary classification
        num_layers=3,
        dropout=0.3
    )
    
    print(f"Model initialized with input_dim={input_dim}")
    
    # Train the model
    trainer = GNNTrainer(model, graph_data)
    final_acc = trainer.train(epochs=200)
    
    print(f"Training completed with accuracy: {final_acc:.4f}")
    
    # Save model architecture info
    model_info = {
        'input_dim': input_dim,
        'hidden_dim': 64,
        'output_dim': 2,
        'num_layers': 3,
        'final_accuracy': float(final_acc)
    }
    
    model_path = get_model_path('model_info.json')
    save_json(model_info, model_path)
    
    return model, graph_data, trainer


def generate_explanations(model, graph_data, df: pd.DataFrame):
    """Step 4: Generate explanations for predictions"""
    print_separator()
    print("STEP 4: GENERATING EXPLANATIONS")
    print_separator()
    
    # Define feature names (should match the features used in the model)
    feature_names = [
        'PatientAge',
        'PatientGender',
        'ClaimAmount',
        'ClaimStatus',
        'PotentialFraud'
    ]
    
    # GNN explainer
    explainer = GNNExplainer(model, graph_data, feature_names)
    
    # Explain a few sample nodes
    explanations = []
    num_samples = min(5, graph_data.x.shape[0])
    
    for i in range(num_samples):
        try:
            # Generate explanation report for a node
            report = explainer.generate_explanation_report(
                i, 
                output_dir=str(get_output_path('', 'explanations'))
            )
            explanations.append({
                'node_idx': i,
                'explanation': f"Node {i} analysis completed"
            })
            print(f"\nExplanation for node {i} completed")
        except Exception as e:
            print(f"Warning: Could not generate explanation for node {i}: {str(e)}")
            explanations.append({
                'node_idx': i,
                'explanation': f"Error: {str(e)}"
            })
    
    # Save explanations
    output_path = get_output_path('explanations.json', 'explanations')
    save_json({'explanations': explanations}, output_path)
    
    return explanations


def main():
    """Main pipeline execution"""
    print("=" * 60)
    print("HEALTHCARE FRAUD DETECTION PIPELINE")
    print("Knowledge Graph + Graph Neural Network")
    print("=" * 60)
    
    try:
        # Step 1: Preprocessing
        cleaned_data = run_preprocessing()
        
        # Step 2: Build Knowledge Graph
        kg_builder, graph_data = build_knowledge_graph(cleaned_data)
        
        # Step 3: Train Model
        model, graph_data, trainer = train_model(graph_data, cleaned_data)
        
        # Step 4: Generate Explanations
        explanations = generate_explanations(model, graph_data, cleaned_data)
        
        print_separator()
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print_separator()
        print("\nOutput files saved to:")
        print(f"  - Processed data: {get_data_path('cleaned_healthcare_data.csv', 'processed')}")
        print(f"  - Model info: {get_model_path('model_info.json')}")
        print(f"  - Explanations: {get_output_path('explanations.json', 'explanations')}")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

