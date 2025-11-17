import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import torch
import shap
from lime.lime_tabular import LimeTabularExplainer

# Import custom modules
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.preprocessing import HealthcareDataPreprocessor
from src.kg_gnn_model import HealthcareKnowledgeGraph, HealthcareGNN, GNNTrainer

# Page configuration
st.set_page_config(
    page_title="Healthcare Claims Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'kg' not in st.session_state:
    st.session_state.kg = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Main header
st.markdown('<h1 class="main-header">🏥 Healthcare Claims Knowledge Graph & GNN Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2382/2382461.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Select Module",
        ["📤 Data Upload", "🔧 Preprocessing", "🕸️ Knowledge Graph", "🤖 GNN Analysis", "📊 Insights & Explainability"]
    )
    
    st.markdown("---")
    st.subheader("System Status")
    
    if st.session_state.data_loaded:
        st.success("✅ Data Loaded")
    else:
        st.warning("⏳ No Data")
    
    if st.session_state.kg is not None:
        st.success("✅ KG Built")
    else:
        st.info("⏳ KG Pending")
    
    if st.session_state.model is not None:
        st.success("✅ Model Trained")
    else:
        st.info("⏳ Model Pending")

# Page 1: Data Upload
if page == "📤 Data Upload":
    st.header("📤 Upload Healthcare Claims Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload healthcare claims dataset in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_raw = df
                st.session_state.data_loaded = True
                
                st.success(f"✅ Successfully loaded {len(df)} records!")
                
                # Preview data
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Basic statistics
                st.subheader("Dataset Overview")
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Total Records", f"{len(df):,}")
                with col_b:
                    st.metric("Columns", len(df.columns))
                with col_c:
                    st.metric("Unique Patients", df['PatientID'].nunique())
                with col_d:
                    st.metric("Unique Providers", df['ProviderID'].nunique())
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        st.info("""
        **Expected Columns:**
        - ClaimID
        - PatientID
        - ProviderID
        - ClaimAmount
        - ClaimDate
        - DiagnosisCode
        - ProcedureCode
        - ClaimStatus
        - PatientAge
        - PatientGender
        - ProviderSpecialty
        """)

# Page 2: Preprocessing
elif page == "🔧 Preprocessing":
    st.header("🔧 Data Cleaning & Preprocessing")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    else:
        if st.button("🚀 Start Preprocessing", type="primary"):
            with st.spinner("Processing data..."):
                # Create preprocessor
                df_raw = st.session_state.df_raw
                
                # Save temporarily to a proper location
                import tempfile
                import os
                temp_dir = tempfile.gettempdir()
                temp_file = os.path.join(temp_dir, 'healthcare_temp_data.csv')
                df_raw.to_csv(temp_file, index=False)
                
                try:
                    # Preprocess
                    preprocessor = HealthcareDataPreprocessor(temp_file)
                    df_cleaned = preprocessor.preprocess()
                    
                    st.session_state.df_cleaned = df_cleaned
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
            st.success("✅ Preprocessing completed!")
        
        if st.session_state.df_cleaned is not None:
            df = st.session_state.df_cleaned
            
            # Display metrics
            st.subheader("Preprocessing Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Records After Cleaning", f"{len(df):,}")
            with col2:
                duplicates_removed = len(st.session_state.df_raw) - len(df)
                st.metric("Duplicates Removed", duplicates_removed)
            with col3:
                fraud_count = df['PotentialFraud'].sum()
                st.metric("Potential Fraud Cases", fraud_count)
            with col4:
                avg_amount = df['ClaimAmount'].mean()
                st.metric("Avg Claim Amount", f"${avg_amount:,.2f}")
            
            # Visualizations
            st.subheader("Data Distributions")
            
            tab1, tab2, tab3 = st.tabs(["Claim Amounts", "Status Distribution", "Age Distribution"])
            
            with tab1:
                fig = px.histogram(df, x='ClaimAmount', nbins=50,
                                 title="Distribution of Claim Amounts",
                                 labels={'ClaimAmount': 'Claim Amount ($)'})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                status_counts = df['ClaimStatus'].value_counts()
                fig = px.pie(values=status_counts.values, names=status_counts.index,
                           title="Claim Status Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig = px.histogram(df, x='PatientAge', nbins=30,
                                 title="Patient Age Distribution",
                                 labels={'PatientAge': 'Age'})
                st.plotly_chart(fig, use_container_width=True)

# Page 3: Knowledge Graph
elif page == "🕸️ Knowledge Graph":
    st.header("🕸️ Knowledge Graph Construction & Analysis")
    
    if st.session_state.df_cleaned is None:
        st.warning("⚠️ Please complete preprocessing first!")
    else:
        if st.button("🔨 Build Knowledge Graph", type="primary"):
            with st.spinner("Building knowledge graph..."):
                df = st.session_state.df_cleaned
                
                # Build KG
                kg = HealthcareKnowledgeGraph(df)
                G = kg.build_graph()
                
                st.session_state.kg = kg
                st.session_state.G = G
                
            st.success("✅ Knowledge graph built successfully!")
        
        if st.session_state.kg is not None:
            G = st.session_state.G
            
            # Graph statistics
            st.subheader("Graph Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Nodes", G.number_of_nodes())
            with col2:
                st.metric("Total Edges", G.number_of_edges())
            with col3:
                density = nx.density(G)
                st.metric("Graph Density", f"{density:.4f}")
            with col4:
                avg_clustering = nx.average_clustering(G)
                st.metric("Avg Clustering", f"{avg_clustering:.4f}")
            
            # Node type distribution
            st.subheader("Node Type Distribution")
            node_types = [G.nodes[node]['node_type'] for node in G.nodes()]
            type_counts = pd.Series(node_types).value_counts()
            
            fig = px.bar(x=type_counts.index, y=type_counts.values,
                        labels={'x': 'Node Type', 'y': 'Count'},
                        title="Distribution of Node Types")
            st.plotly_chart(fig, use_container_width=True)
            
            # Degree distribution
            st.subheader("Degree Distribution")
            degrees = [G.degree(node) for node in G.nodes()]
            
            fig = px.histogram(x=degrees, nbins=50,
                             labels={'x': 'Degree', 'y': 'Frequency'},
                             title="Node Degree Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Network visualization sample
            st.subheader("Network Visualization (Sample)")
            st.info("Showing subgraph of 50 most connected nodes")
            
            # Get top nodes by degree
            top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:50]
            top_node_ids = [node[0] for node in top_nodes]
            subG = G.subgraph(top_node_ids)
            
            # Create visualization
            pos = nx.spring_layout(subG, k=0.5, iterations=50)
            
            edge_trace = []
            for edge in subG.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace.append(
                    go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                             mode='lines',
                             line=dict(width=0.5, color='#888'),
                             hoverinfo='none',
                             showlegend=False)
                )
            
            node_x = [pos[node][0] for node in subG.nodes()]
            node_y = [pos[node][1] for node in subG.nodes()]
            node_colors = [subG.nodes[node]['node_type'] for node in subG.nodes()]
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    size=10,
                    color=node_colors,
                    colorscale='Viridis',
                    line=dict(width=2, color='white')
                )
            )
            
            fig = go.Figure(data=edge_trace + [node_trace],
                          layout=go.Layout(
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=0, l=0, r=0, t=0),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              height=600
                          ))
            
            st.plotly_chart(fig, use_container_width=True)

# Page 4: GNN Analysis
elif page == "🤖 GNN Analysis":
    st.header("🤖 Graph Neural Network Training & Prediction")
    
    if st.session_state.kg is None:
        st.warning("⚠️ Please build knowledge graph first!")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Model Configuration")
            
            hidden_dim = st.slider("Hidden Dimension", 32, 256, 64, 32)
            num_layers = st.slider("Number of Layers", 2, 5, 3)
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.3, 0.1)
            epochs = st.slider("Training Epochs", 50, 500, 200, 50)
            
        with col2:
            st.info("""
            **Model Architecture:**
            - GCN (Graph Convolutional Network)
            - Task: Fraud Detection
            - Output: Binary Classification
            """)
        
        if st.button("🚀 Train GNN Model", type="primary"):
            with st.spinner("Training model..."):
                df = st.session_state.df_cleaned
                kg = st.session_state.kg
                
                # Convert to PyTorch Geometric
                data = kg.to_pytorch_geometric(df)
                
                # Initialize model
                model = HealthcareGNN(
                    input_dim=data.num_node_features,
                    hidden_dim=hidden_dim,
                    output_dim=2,
                    num_layers=num_layers,
                    dropout=dropout
                )
                
                # Train
                trainer = GNNTrainer(model, data)
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                final_acc = trainer.train(epochs=epochs)
                
                progress_bar.progress(100)
                
                # Make predictions
                predictions, probabilities = trainer.predict()
                
                st.session_state.model = model
                st.session_state.predictions = predictions
                st.session_state.probabilities = probabilities
                
            st.success(f"✅ Training completed! Final Accuracy: {final_acc:.4f}")
        
        if st.session_state.predictions is not None:
            st.subheader("Prediction Results")
            
            predictions = st.session_state.predictions
            probabilities = st.session_state.probabilities
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fraud_detected = predictions.sum()
                st.metric("Fraud Cases Detected", fraud_detected)
            
            with col2:
                fraud_pct = (fraud_detected / len(predictions)) * 100
                st.metric("Fraud Percentage", f"{fraud_pct:.2f}%")
            
            with col3:
                avg_confidence = probabilities.max(axis=1).mean()
                st.metric("Avg Confidence", f"{avg_confidence:.4f}")
            
            # Confidence distribution
            st.subheader("Prediction Confidence Distribution")
            
            max_probs = probabilities.max(axis=1)
            fig = px.histogram(x=max_probs, nbins=50,
                             labels={'x': 'Confidence Score', 'y': 'Frequency'},
                             title="Model Confidence Distribution")
            st.plotly_chart(fig, use_container_width=True)

# Page 5: Insights & Explainability
elif page == "📊 Insights & Explainability":
    st.header("📊 Business Insights & Model Explainability")
    
    if st.session_state.df_cleaned is None:
        st.warning("⚠️ No data available for analysis!")
    else:
        df = st.session_state.df_cleaned
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Business Metrics",
            "🔍 Fraud Analysis",
            "🏥 Provider Insights",
            "🤖 Model Explainability"
        ])
        
        with tab1:
            st.subheader("Key Business Metrics")
            
            # Financial metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_claims = df['ClaimAmount'].sum()
                st.metric("Total Claims Value", f"${total_claims:,.2f}")
            
            with col2:
                approved_value = df[df['ClaimStatus'] == 'Approved']['ClaimAmount'].sum()
                st.metric("Approved Claims", f"${approved_value:,.2f}")
            
            with col3:
                pending_value = df[df['ClaimStatus'] == 'Pending']['ClaimAmount'].sum()
                st.metric("Pending Claims", f"${pending_value:,.2f}")
            
            with col4:
                denied_value = df[df['ClaimStatus'] == 'Denied']['ClaimAmount'].sum()
                st.metric("Denied Claims", f"${denied_value:,.2f}")
            
            # Trends over time
            st.subheader("Claims Trends Over Time")
            
            if 'ClaimDate' in df.columns:
                df['ClaimDate'] = pd.to_datetime(df['ClaimDate'])
                monthly_claims = df.groupby(df['ClaimDate'].dt.to_period('M'))['ClaimAmount'].agg(['sum', 'count'])
                monthly_claims.index = monthly_claims.index.to_timestamp()
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=monthly_claims.index, y=monthly_claims['sum'],
                             name="Claim Value", line=dict(color='blue')),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=monthly_claims.index, y=monthly_claims['count'],
                             name="Claim Count", line=dict(color='red')),
                    secondary_y=True
                )
                
                fig.update_layout(title="Monthly Claims Trends", height=400)
                fig.update_xaxes(title_text="Month")
                fig.update_yaxes(title_text="Total Value ($)", secondary_y=False)
                fig.update_yaxes(title_text="Count", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Fraud Detection Analysis")
            
            fraud_df = df[df['PotentialFraud'] == True]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Fraud Cases", len(fraud_df))
                st.metric("Fraud Amount", f"${fraud_df['ClaimAmount'].sum():,.2f}")
            
            with col2:
                fraud_rate = (len(fraud_df) / len(df)) * 100
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                avg_fraud_amount = fraud_df['ClaimAmount'].mean()
                st.metric("Avg Fraud Amount", f"${avg_fraud_amount:,.2f}")
            
            # Fraud by specialty
            st.subheader("Fraud Cases by Provider Specialty")
            
            fraud_specialty = fraud_df['ProviderSpecialty'].value_counts().head(10)
            fig = px.bar(x=fraud_specialty.index, y=fraud_specialty.values,
                        labels={'x': 'Specialty', 'y': 'Fraud Cases'},
                        title="Top 10 Specialties with Fraud Cases")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Provider Performance Analysis")
            
            # Top providers by claim volume
            top_providers = df.groupby('ProviderID').agg({
                'ClaimAmount': ['sum', 'count', 'mean'],
                'ClaimStatus': lambda x: (x == 'Approved').sum() / len(x)
            }).round(2)
            
            top_providers.columns = ['Total Amount', 'Claim Count', 'Avg Amount', 'Approval Rate']
            top_providers = top_providers.sort_values('Total Amount', ascending=False).head(10)
            
            st.dataframe(top_providers, use_container_width=True)
            
            # Specialty performance
            st.subheader("Performance by Specialty")
            
            specialty_perf = df.groupby('ProviderSpecialty').agg({
                'ClaimAmount': 'mean',
                'ClaimStatus': lambda x: (x == 'Approved').sum() / len(x) * 100
            }).round(2)
            specialty_perf.columns = ['Avg Claim Amount', 'Approval Rate (%)']
            specialty_perf = specialty_perf.sort_values('Avg Claim Amount', ascending=False).head(10)
            
            fig = px.bar(specialty_perf, x=specialty_perf.index, y='Avg Claim Amount',
                        title="Average Claim Amount by Specialty",
                        labels={'x': 'Specialty', 'Avg Claim Amount': 'Amount ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Model Explainability")
            
            if st.session_state.model is not None:
                st.info("""
                **SHAP & LIME Explanations**
                
                These tools help understand which features contribute most to fraud predictions:
                - **SHAP**: Shows global feature importance
                - **LIME**: Provides local explanations for individual predictions
                """)
                
                st.success("Model explainability features available after training!")
            else:
                st.warning("Train the GNN model first to enable explainability features.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Healthcare Claims Knowledge Graph & GNN Analysis System</p>
    <p>Built with Streamlit • PyTorch Geometric • NetworkX • Plotly</p>
</div>
""", unsafe_allow_html=True)