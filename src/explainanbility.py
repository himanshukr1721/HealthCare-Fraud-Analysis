import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import seaborn as sns

class GNNExplainer:
    """
    Explainability toolkit for GNN models using SHAP and LIME
    """
    
    def __init__(self, model, data, feature_names):
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.device = next(model.parameters()).device
        
    def get_node_embeddings(self):
        """Extract node embeddings from trained model"""
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embeddings(
                self.data.x.to(self.device),
                self.data.edge_index.to(self.device)
            )
        return embeddings.cpu().numpy()
    
    def compute_feature_importance(self):
        """Compute feature importance using gradient-based method"""
        self.model.eval()
        
        # Enable gradients for input
        x = self.data.x.clone().requires_grad_(True).to(self.device)
        edge_index = self.data.edge_index.to(self.device)
        
        # Forward pass
        output = self.model(x, edge_index)
        
        # Get predictions for fraud class (class 1)
        fraud_scores = output[:, 1]
        
        # Compute gradients
        self.model.zero_grad()
        fraud_scores.sum().backward()
        
        # Feature importance is the absolute gradient
        importance = torch.abs(x.grad).mean(dim=0).cpu().numpy()
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def explain_with_lime(self, node_idx, num_features=5):
        """
        Explain individual node prediction using LIME
        
        Args:
            node_idx: Index of node to explain
            num_features: Number of top features to show
        """
        # Prepare data for LIME
        X_train = self.data.x.cpu().numpy()
        
        # Create predict function for LIME
        def predict_fn(X):
            self.model.eval()
            with torch.no_grad():
                # Create batch of predictions
                predictions = []
                for sample in X:
                    # Replace target node features
                    x_temp = self.data.x.clone()
                    x_temp[node_idx] = torch.FloatTensor(sample)
                    
                    # Predict
                    output = self.model(
                        x_temp.to(self.device),
                        self.data.edge_index.to(self.device)
                    )
                    probs = torch.exp(output[node_idx]).cpu().numpy()
                    predictions.append(probs)
                
                return np.array(predictions)
        
        # Initialize LIME explainer
        explainer = LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=['Non-Fraud', 'Fraud'],
            mode='classification'
        )
        
        # Explain instance
        explanation = explainer.explain_instance(
            X_train[node_idx],
            predict_fn,
            num_features=num_features
        )
        
        return explanation
    
    def explain_with_shap(self, background_samples=100):
        """
        Compute SHAP values for model explanations
        
        Args:
            background_samples: Number of background samples for SHAP
        """
        # Get background dataset
        background_indices = np.random.choice(
            len(self.data.x),
            size=min(background_samples, len(self.data.x)),
            replace=False
        )
        background = self.data.x[background_indices].cpu().numpy()
        
        # Define prediction function for SHAP
        def predict_fn(X):
            self.model.eval()
            with torch.no_grad():
                predictions = []
                for sample in X:
                    # Create temporary data with sample
                    x_temp = torch.FloatTensor(sample).unsqueeze(0)
                    
                    # For simplicity, use mean of all nodes' predictions
                    # In practice, you'd want node-specific predictions
                    output = self.model(
                        self.data.x.to(self.device),
                        self.data.edge_index.to(self.device)
                    )
                    probs = torch.exp(output).mean(dim=0).cpu().numpy()
                    predictions.append(probs)
                
                return np.array(predictions)
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Compute SHAP values for test samples
        test_samples = self.data.x[:100].cpu().numpy()
        shap_values = explainer.shap_values(test_samples)
        
        return shap_values, test_samples
    
    def plot_feature_importance(self, importance_df, top_n=10, save_path=None):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance (Gradient-based)')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_lime_explanation(self, explanation, save_path=None):
        """Visualize LIME explanation"""
        fig = explanation.as_pyplot_figure()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_shap_summary(self, shap_values, test_samples, save_path=None):
        """Plot SHAP summary"""
        plt.figure(figsize=(10, 8))
        
        # SHAP summary plot for fraud class (class 1)
        shap.summary_plot(
            shap_values[1],
            test_samples,
            feature_names=self.feature_names,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def analyze_node_neighborhood(self, node_idx, hops=2):
        """
        Analyze the neighborhood of a node to understand its context
        
        Args:
            node_idx: Index of node to analyze
            hops: Number of hops to consider
        """
        import networkx as nx
        
        # Convert edge_index to NetworkX graph
        edge_list = self.data.edge_index.t().cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(edge_list)
        
        # Get k-hop neighborhood
        neighbors = nx.single_source_shortest_path_length(G, node_idx, cutoff=hops)
        
        neighbor_features = self.data.x[list(neighbors.keys())].cpu().numpy()
        neighbor_predictions = self.get_predictions_for_nodes(list(neighbors.keys()))
        
        analysis = {
            'node_idx': node_idx,
            'num_neighbors': len(neighbors),
            'neighbor_features_mean': neighbor_features.mean(axis=0),
            'neighbor_features_std': neighbor_features.std(axis=0),
            'neighbor_fraud_rate': (neighbor_predictions == 1).mean()
        }
        
        return analysis, neighbors
    
    def get_predictions_for_nodes(self, node_indices):
        """Get predictions for specific nodes"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                self.data.x.to(self.device),
                self.data.edge_index.to(self.device)
            )
            predictions = output.argmax(dim=1)[node_indices].cpu().numpy()
        
        return predictions
    
    def generate_explanation_report(self, node_idx, output_dir='explanations'):
        """
        Generate comprehensive explanation report for a node
        
        Args:
            node_idx: Node to explain
            output_dir: Directory to save visualizations
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
        
        # 2. Model prediction
        prediction = self.get_predictions_for_nodes([node_idx])[0]
        print(f"Model Prediction: {'Fraud' if prediction == 1 else 'Non-Fraud'}")
        
        # 3. Feature importance
        print("\n" + "-"*60)
        print("FEATURE IMPORTANCE")
        print("-"*60)
        
        importance_df = self.compute_feature_importance()
        print(importance_df.head(10))
        
        fig = self.plot_feature_importance(importance_df)
        fig.savefig(f"{output_dir}/feature_importance.png")
        plt.close()
        
        # 4. LIME explanation
        print("\n" + "-"*60)
        print("LIME EXPLANATION")
        print("-"*60)
        
        lime_exp = self.explain_with_lime(node_idx)
        print("\nLIME Feature Weights:")
        for feat, weight in lime_exp.as_list():
            print(f"  {feat}: {weight:.4f}")
        
        fig = self.plot_lime_explanation(lime_exp)
        fig.savefig(f"{output_dir}/lime_explanation_node_{node_idx}.png")
        plt.close()
        
        # 5. Neighborhood analysis
        print("\n" + "-"*60)
        print("NEIGHBORHOOD ANALYSIS")
        print("-"*60)
        
        analysis, neighbors = self.analyze_node_neighborhood(node_idx)
        print(f"\nNumber of neighbors (2-hops): {analysis['num_neighbors']}")
        print(f"Neighbor fraud rate: {analysis['neighbor_fraud_rate']:.2%}")
        
        print("\n" + "="*60)
        print("REPORT GENERATION COMPLETE")
        print(f"Visualizations saved to: {output_dir}/")
        print("="*60 + "\n")
        
        return {
            'node_idx': node_idx,
            'features': node_features,
            'true_label': node_label,
            'prediction': prediction,
            'importance': importance_df,
            'lime_explanation': lime_exp,
            'neighborhood_analysis': analysis
        }


# Usage Example
if __name__ == "__main__":
    # Assuming model and data are already loaded
    from src.kg_gnn_model import HealthcareGNN, HealthcareKnowledgeGraph
    import torch
    
    # Load your trained model and data
    # model = HealthcareGNN(...)
    # data = ...
    
    # Define feature names
    feature_names = [
        'PatientAge',
        'PatientGender',
        'ClaimAmount',
        'ClaimStatus',
        'PotentialFraud'
    ]
    
    # Initialize explainer
    # explainer = GNNExplainer(model, data, feature_names)
    
    # Generate explanation for specific node
    # node_to_explain = 42
    # report = explainer.generate_explanation_report(node_to_explain)
    
    # Compute global feature importance
    # importance = explainer.compute_feature_importance()
    # explainer.plot_feature_importance(importance)
    
    # Compute SHAP values
    # shap_values, test_samples = explainer.explain_with_shap()
    # explainer.plot_shap_summary(shap_values, test_samples)
    
    print("Explainability module loaded successfully!")