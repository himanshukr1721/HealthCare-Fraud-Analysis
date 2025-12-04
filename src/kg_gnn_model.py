import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class HealthcareKnowledgeGraph:
    """
    Build Knowledge Graph from healthcare claims data
    """
    
    def __init__(self, df):
        self.df = df
        self.G = nx.Graph()
        self.node_mappings = {}
        self.edge_index = None
        self.node_features = None
        self.edge_attr = None
        
    def create_nodes(self):
        """Create nodes for patients, providers, diagnoses, procedures"""
        print("Creating knowledge graph nodes...")
        
        # Patient nodes
        patients = self.df['PatientID'].unique()
        for idx, patient in enumerate(patients):
            self.G.add_node(f"patient_{patient}", 
                          node_type='patient',
                          node_id=patient)
            self.node_mappings[f"patient_{patient}"] = idx
            
        # Provider nodes
        offset = len(patients)
        providers = self.df['ProviderID'].unique()
        for idx, provider in enumerate(providers):
            self.G.add_node(f"provider_{provider}",
                          node_type='provider',
                          node_id=provider)
            self.node_mappings[f"provider_{provider}"] = offset + idx
            
        # Diagnosis nodes
        offset += len(providers)
        diagnoses = self.df['DiagnosisCode'].unique()
        for idx, diagnosis in enumerate(diagnoses):
            self.G.add_node(f"diagnosis_{diagnosis}",
                          node_type='diagnosis',
                          node_id=diagnosis)
            self.node_mappings[f"diagnosis_{diagnosis}"] = offset + idx
            
        # Procedure nodes
        offset += len(diagnoses)
        procedures = self.df['ProcedureCode'].unique()
        for idx, procedure in enumerate(procedures):
            self.G.add_node(f"procedure_{procedure}",
                          node_type='procedure',
                          node_id=procedure)
            self.node_mappings[f"procedure_{procedure}"] = offset + idx
            
        print(f"Created {self.G.number_of_nodes()} nodes")
        
    def create_edges(self):
        """Create edges representing relationships"""
        print("Creating knowledge graph edges...")
        
        for _, row in self.df.iterrows():
            patient = f"patient_{row['PatientID']}"
            provider = f"provider_{row['ProviderID']}"
            diagnosis = f"diagnosis_{row['DiagnosisCode']}"
            procedure = f"procedure_{row['ProcedureCode']}"
            
            # Patient-Provider relationship
            self.G.add_edge(patient, provider,
                          edge_type='visited',
                          claim_amount=row['ClaimAmount'],
                          claim_status=row['ClaimStatus'])
            
            # Patient-Diagnosis relationship
            self.G.add_edge(patient, diagnosis,
                          edge_type='has_diagnosis',
                          claim_amount=row['ClaimAmount'])
            
            # Patient-Procedure relationship
            self.G.add_edge(patient, procedure,
                          edge_type='received_procedure',
                          claim_amount=row['ClaimAmount'])
            
            # Provider-Diagnosis relationship
            self.G.add_edge(provider, diagnosis,
                          edge_type='treats',
                          frequency=1)
            
            # Diagnosis-Procedure relationship
            self.G.add_edge(diagnosis, procedure,
                          edge_type='requires',
                          frequency=1)
                          
        print(f"Created {self.G.number_of_edges()} edges")
        
    def compute_graph_statistics(self):
        """Compute important graph metrics"""
        print("\nGraph Statistics:")
        print(f"Number of nodes: {self.G.number_of_nodes()}")
        print(f"Number of edges: {self.G.number_of_edges()}")
        print(f"Graph density: {nx.density(self.G):.4f}")
        print(f"Average clustering coefficient: {nx.average_clustering(self.G):.4f}")
        
        if nx.is_connected(self.G):
            print(f"Average shortest path: {nx.average_shortest_path_length(self.G):.2f}")
        else:
            largest_cc = max(nx.connected_components(self.G), key=len)
            print(f"Number of connected components: {nx.number_connected_components(self.G)}")
            print(f"Largest component size: {len(largest_cc)}")
            
    def build_graph(self):
        """Build complete knowledge graph"""
        self.create_nodes()
        self.create_edges()
        self.compute_graph_statistics()
        return self.G
    
    def to_pytorch_geometric(self, df):
        """Convert NetworkX graph to PyTorch Geometric format"""
        print("\nConverting to PyTorch Geometric format...")
        
        # Create node feature matrix
        node_features = []
        node_labels = []
        patient_mask = []  # Track which nodes are patients (have labels)
        
        le_type = LabelEncoder()
        le_status = LabelEncoder()
        
        # Encode categorical features
        df['ClaimStatus_encoded'] = le_status.fit_transform(df['ClaimStatus'])
        
        # Build feature matrix for each node
        for node in self.G.nodes():
            node_type = self.G.nodes[node]['node_type']
            
            if node_type == 'patient':
                patient_id = self.G.nodes[node]['node_id']
                patient_data = df[df['PatientID'] == patient_id].iloc[0]
                
                # FIXED: Removed PotentialFraud from features (was causing data leakage)
                features = [
                    patient_data['PatientAge'],
                    1 if patient_data['PatientGender'] == 'Male' else 0,
                    patient_data['ClaimAmount'],
                    patient_data['ClaimStatus_encoded']
                ]
                label = 1 if patient_data['PotentialFraud'] else 0
                patient_mask.append(True)
                
            elif node_type == 'provider':
                provider_id = self.G.nodes[node]['node_id']
                provider_data = df[df['ProviderID'] == provider_id]
                
                # FIXED: Removed PotentialFraud from features (was causing data leakage)
                features = [
                    len(provider_data),  # Number of claims
                    provider_data['ClaimAmount'].mean(),
                    provider_data['ClaimAmount'].sum(),
                    (provider_data['ClaimStatus'] == 'Approved').sum()
                ]
                label = 0
                patient_mask.append(False)
                
            else:  # diagnosis or procedure
                features = [0, 0, 0, 0]
                label = 0
                patient_mask.append(False)
                
            node_features.append(features)
            node_labels.append(label)
        
        # Convert to tensors
        x = torch.FloatTensor(node_features)
        y = torch.LongTensor(node_labels)
        patient_mask = torch.BoolTensor(patient_mask)
        
        # Create edge index
        edge_index = []
        for edge in self.G.edges():
            src = self.node_mappings[edge[0]]
            dst = self.node_mappings[edge[1]]
            edge_index.append([src, dst])
            edge_index.append([dst, src])  # Undirected graph
            
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        data.patient_mask = patient_mask  # Store patient mask for evaluation
        
        # FIXED: Proper graph-aware train/test split (split by patients, not random nodes)
        # This prevents data leakage through graph connections
        patient_ids = df['PatientID'].unique()
        train_patients, temp_patients = train_test_split(
            patient_ids, test_size=0.3, random_state=42, stratify=None
        )
        val_patients, test_patients = train_test_split(
            temp_patients, test_size=0.33, random_state=42
        )
        
        train_patients_set = set(train_patients)
        val_patients_set = set(val_patients)
        test_patients_set = set(test_patients)
        
        num_nodes = data.num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Assign masks based on patient membership
        for idx, node in enumerate(self.G.nodes()):
            node_type = self.G.nodes[node]['node_type']
            if node_type == 'patient':
                patient_id = self.G.nodes[node]['node_id']
                if patient_id in train_patients_set:
                    train_mask[idx] = True
                elif patient_id in val_patients_set:
                    val_mask[idx] = True
                elif patient_id in test_patients_set:
                    test_mask[idx] = True
            else:
                # Non-patient nodes (providers, diagnoses, procedures) go to training
                # They don't have labels, so they're only used for graph structure
                train_mask[idx] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        # Diagnostic information
        print(f"Created PyTorch Geometric data with {data.num_nodes} nodes and {data.num_edges} edges")
        print(f"\nLabel Distribution:")
        print(f"  Total nodes: {len(y)}")
        print(f"  Patient nodes (with labels): {patient_mask.sum().item()}")
        print(f"  Class 0 (Legitimate): {(y[patient_mask] == 0).sum().item()}")
        print(f"  Class 1 (Fraud): {(y[patient_mask] == 1).sum().item()}")
        print(f"\nTrain/Val/Test Split (patient nodes only):")
        train_patient_count = (train_mask & patient_mask).sum().item()
        val_patient_count = (val_mask & patient_mask).sum().item()
        test_patient_count = (test_mask & patient_mask).sum().item()
        print(f"  Train patient nodes: {train_patient_count}")
        print(f"  Val patient nodes: {val_patient_count}")
        print(f"  Test patient nodes: {test_patient_count}")
        print(f"  Total train nodes (including non-patients): {train_mask.sum().item()}")
        print(f"  Total val nodes: {val_mask.sum().item()}")
        print(f"  Total test nodes: {test_mask.sum().item()}")
        
        return data


class HealthcareGNN(nn.Module):
    """
    Graph Neural Network for healthcare fraud detection and claim prediction
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.3):
        super(HealthcareGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
    def forward(self, x, edge_index):
        """Forward pass"""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings from penultimate layer"""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        return x


class GNNTrainer:
    """
    Training and evaluation pipeline for GNN
    """
    
    def __init__(self, model, data, device='cpu', use_class_weights=True):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # FIXED: Add class weights for imbalanced data
        if use_class_weights and hasattr(data, 'patient_mask'):
            # Calculate class weights based on patient nodes only
            patient_labels = data.y[data.patient_mask & data.train_mask]
            if len(patient_labels) > 0:
                class_counts = torch.bincount(patient_labels, minlength=2)
                # Avoid division by zero
                class_counts = class_counts.float()
                if class_counts[0] > 0 and class_counts[1] > 0:
                    total = class_counts.sum()
                    class_weights = total / (2.0 * class_counts)
                    class_weights = class_weights / class_weights.sum() * 2.0  # Normalize
                    self.class_weights = class_weights.to(device)
                    print(f"Class weights: {self.class_weights.cpu().numpy()}")
                else:
                    self.class_weights = None
            else:
                self.class_weights = None
        else:
            self.class_weights = None
        
    def train_epoch(self):
        """Train for one epoch using only training nodes"""
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(self.data.x, self.data.edge_index)
        
        # FIXED: Only train on patient nodes (they have labels)
        train_patient_mask = self.data.train_mask & self.data.patient_mask
        if train_patient_mask.sum() > 0:
            if self.class_weights is not None:
                loss = F.nll_loss(
                    out[train_patient_mask], 
                    self.data.y[train_patient_mask],
                    weight=self.class_weights
                )
            else:
                loss = F.nll_loss(out[train_patient_mask], self.data.y[train_patient_mask])
        else:
            # Fallback if no patient nodes in training
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, mask=None):
        """Evaluate model on specified mask (train/val/test) - only on patient nodes"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
            
            if mask is None:
                # FIXED: Only evaluate on patient nodes by default
                if hasattr(self.data, 'patient_mask'):
                    mask = self.data.patient_mask
                else:
                    mask = torch.ones(self.data.num_nodes, dtype=torch.bool, device=self.device)
            else:
                # FIXED: Combine provided mask with patient mask (only evaluate patients)
                if hasattr(self.data, 'patient_mask'):
                    mask = mask & self.data.patient_mask
            
            if mask.sum() > 0:
                correct = (pred[mask] == self.data.y[mask]).sum().item()
                accuracy = correct / mask.sum().item()
            else:
                accuracy = 0.0
            
        return accuracy
    
    def train(self, epochs=200):
        """Full training loop with train/val/test evaluation"""
        print("\nTraining GNN model...")
        
        # FIXED: Show patient node counts for clarity
        if hasattr(self.data, 'patient_mask'):
            train_patient_count = (self.data.train_mask & self.data.patient_mask).sum().item()
            val_patient_count = (self.data.val_mask & self.data.patient_mask).sum().item()
            test_patient_count = (self.data.test_mask & self.data.patient_mask).sum().item()
            print(f"Training on {train_patient_count} patient nodes")
            print(f"Validating on {val_patient_count} patient nodes")
            print(f"Testing on {test_patient_count} patient nodes")
        else:
            print(f"Train nodes: {self.data.train_mask.sum().item()}, Val nodes: {self.data.val_mask.sum().item()}, Test nodes: {self.data.test_mask.sum().item()}")
        
        best_val_acc = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            loss = self.train_epoch()
            
            if (epoch + 1) % 20 == 0:
                # FIXED: Evaluate only on patient nodes
                train_acc = self.evaluate(self.data.train_mask)
                val_acc = self.evaluate(self.data.val_mask)
                test_acc = self.evaluate(self.data.test_mask)
                
                print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Early stopping if validation doesn't improve
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} evaluations)")
                    break
        
        # Final evaluation
        final_train_acc = self.evaluate(self.data.train_mask)
        final_val_acc = self.evaluate(self.data.val_mask)
        final_test_acc = self.evaluate(self.data.test_mask)
        
        print(f'\nFinal Results (on patient nodes only):')
        print(f'  Train Accuracy: {final_train_acc:.4f}')
        print(f'  Val Accuracy: {final_val_acc:.4f}')
        print(f'  Test Accuracy: {final_test_acc:.4f}')
        
        return final_test_acc
    
    def predict(self):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
            probs = torch.exp(out)
            
        return pred.cpu().numpy(), probs.cpu().numpy()


# Usage Example
if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('cleaned_healthcare_data.csv')
    
    # Build Knowledge Graph
    kg = HealthcareKnowledgeGraph(df)
    G = kg.build_graph()
    
    # Convert to PyTorch Geometric
    data = kg.to_pytorch_geometric(df)
    
    # Initialize GNN model
    model = HealthcareGNN(
        input_dim=data.num_node_features,
        hidden_dim=64,
        output_dim=2,  # Binary classification: fraud or not
        num_layers=3
    )
    
    # Train model
    trainer = GNNTrainer(model, data)
    trainer.train(epochs=200)
    
    # Make predictions
    predictions, probabilities = trainer.predict()
    
    print("\n✓ GNN training complete!")
    print(f"Detected {predictions.sum()} potential fraud cases")