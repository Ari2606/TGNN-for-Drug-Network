📊 Temporal GNN for Drug Supply Network Analysis
📌 Overview
This project implements a Temporal Graph Neural Network (TGNN) to analyze dark-web drug supply networks (Agora Marketplace, 2014–2015).
It models how vendors interact over time and predicts:
Vulnerability → which vendors are likely to disappear
Adaptation → how the market structure changes
Route shifts → changes in supply connections
🚀 Key Features
🔹 Data Processing
Cleans raw marketplace data
Normalizes categories (Cannabis, Cocaine, etc.)
Handles price, ratings, and missing values
Assigns quarter-based temporal snapshots
🔹 Feature Engineering
For each vendor per quarter:
Listings count
Price stats (mean, median, std, max)
Ratings
Geographic diversity
Category diversity
Route entropy
🔹 Graph Construction
Nodes → Vendors
Edges → Shared:
Destination
Origin
Category
Weighted edges using interaction strength
🔹 Temporal Modeling
Uses GraphSAGE + GRU (memory)
Tracks evolution across quarters
Multi-task learning:
Vulnerability prediction
Adaptation prediction
Route prediction
🧠 Model Architecture
Graph Encoder: GraphSAGE (3 layers)
Temporal Memory: GRU
Heads:
Vulnerability (node-level)
Adaptation (graph-level)
Route shift (edge-level)
📈 Outputs
All results are saved to OUT_DIR:
📊 Data
cleaned_agora.csv
📉 Plots
loss_curves.png
adaptation_over_time.png
top_vulnerable_nodes.png
top_route_shifts.png
network_vulnerability.png
temporal_vulnerability.png
disruption_comparison.png
🌐 Interactive Visualizations
interactive_network.html → network graph
interactive_globe.html → 🌍 global trade routes
📄 Tables
vulnerability_scores_last_quarter.csv
route_shift_scores_last_quarter.csv
adaptation_scores_over_time.csv
💾 Model
temporal_gnn_drug_network.pt
⚙️ How to Run
python drug_network_gnn.py
📂 Requirements
Install dependencies:
pip install numpy pandas torch torch-geometric matplotlib networkx plotly pyvis scikit-learn
📍 Dataset
Input file: Agora.csv
Contains:
Seller
Category
Price
Shipping origin/destination
Ratings
🌍 Visualization Highlights
📌 Vulnerable vendors highlighted
📌 Temporal evolution of risk
📌 Network disruption simulations
📌 Interactive Earth-style globe routes
🧪 Evaluation Metrics
MSE / MAE → vulnerability & adaptation
Binary Cross Entropy → route prediction
F1-score / Accuracy → edge prediction
💡 Use Cases
Dark web market analysis
Supply chain vulnerability detection
Network disruption simulation
Temporal graph learning research
🧠 Novelty
Combines:
Temporal GNN
Multi-task prediction
Network disruption analysis
Interactive globe visualization
📌 Notes
Uses self-supervised labels (no manual annotation)
Works on quarterly snapshots
Designed for research + visualization
