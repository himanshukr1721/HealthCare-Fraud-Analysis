
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.explain.metric import fidelity

class HealthcareGNNExplainer:
    """
    Explainability toolkit for GNN models using GNNExplainer from PyTorch Geometric
    """
    
    def __init__(self, model, data, feature_names):
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.device = next(model.parameters()).device
        
        # Configure the explainer
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
        )
        
    def explain_node(self, node_idx, target=None):
        """
        Explain prediction for a specific node
        
        Args:
            node_idx: Index of node to explain
            target: Target class index (optional, defaults to predicted class)
        """
        self.model.eval()
        
        # Get prediction if target not provided
        if target is None:
            with torch.no_grad():
                out = self.model(self.data.x.to(self.device), self.data.edge_index.to(self.device))
                target = out.argmax(dim=1)[node_idx].item()
                
        # Generate explanation
        explanation = self.explainer(
            x=self.data.x.to(self.device),
            edge_index=self.data.edge_index.to(self.device),
            index=node_idx,
            target=torch.tensor([target], device=self.device)
        )
        
        return explanation
    
    def visualize_explanation(self, explanation, path=None):
        """Visualize the explanation"""
        # Feature importance
        if explanation.node_mask is not None:
            node_mask = explanation.node_mask
            # Aggregate feature importance (e.g., mean across nodes if mask is per-node-per-feature)
            # For 'attributes' mask type, it's usually [num_nodes, num_features] or [1, num_features] depending on implementation details
            # In PyG GNNExplainer for node tasks, it often returns mask for the computation subgraph
            
            # Simplified visualization of top features
            feature_importance = node_mask.mean(dim=0).cpu().numpy()
            
            plt.figure(figsize=(10, 6))
            indices = np.argsort(feature_importance)[-10:]  # Top 10
            plt.barh(range(len(indices)), feature_importance[indices])
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importance (GNNExplainer)')
            
            if path:
                plt.savefig(f"{path}_features.png", bbox_inches='tight')
                plt.close()
            else:
                plt.show()

        # Subgraph visualization is more complex and might require networkx conversion
        # PyG has explanation.visualize_graph() but it might need matplotlib backend adjustment
        
        return explanation

    def generate_explanation_report(self, node_idx, output_dir='explanations'):
        """
        Generate comprehensive explanation report for a node
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"EXPLANATION REPORT FOR NODE {node_idx}")
        print(f"{'='*60}\n")
        
        # 1. Node features
        node_features = self.data.x[node_idx].cpu().numpy()
        node_label = self.data.y[node_idx].item()
        
        print("Node Features:")
        for i, (feat, val) in enumerate(zip(self.feature_names, node_features)):
            print(f"  {feat}: {val:.4f}")
            
        print(f"\nTrue Label: {'Fraud' if node_label == 1 else 'Non-Fraud'}")
        
        # 2. Generate Explanation
        explanation = self.explain_node(node_idx)
        
        # 3. Visualize
        self.visualize_explanation(explanation, path=f"{output_dir}/node_{node_idx}")
        
        print(f"Explanation generated and saved to {output_dir}/node_{node_idx}_features.png")
        
        return explanation

# Usage Example
if __name__ == "__main__":
    # Assuming model and data are already loaded
    # from src.kg_gnn_model import HealthcareGNN, HealthcareKnowledgeGraph
    
    print("GNNExplainer module loaded successfully!")
