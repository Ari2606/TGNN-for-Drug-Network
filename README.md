# 📊 Temporal GNN for Drug Supply Network Analysis

## 📌 Overview
This project implements a **Temporal Graph Neural Network (TGNN)** to analyze dark-web drug supply networks (Agora dataset).

It predicts:
- **Vulnerability** — which vendors may disappear  
- **Adaptation** — how the market evolves  
- **Route Shifts** — changes in supply connections  

---

## 🚀 Features

### 🔹 Data Processing
- Cleans raw dataset  
- Normalizes drug categories  
- Handles missing values  
- Assigns quarterly timestamps  

### 🔹 Feature Engineering
For each vendor:
- Listings count  
- Price statistics (mean, median, std, max)  
- Ratings  
- Geographic diversity  
- Category diversity  
- Route entropy  

### 🔹 Graph Construction
- Nodes = Vendors  
- Edges = Shared origin, destination, or category  
- Weighted connections  

### 🔹 Temporal Modeling
- GraphSAGE + GRU  
- Tracks evolution over time  
- Multi-task learning:
  - Vulnerability prediction  
  - Adaptation prediction  
  - Route prediction  

---

## 🧠 Model Architecture
- **Graph Encoder**: GraphSAGE  
- **Temporal Memory**: GRU  
- **Prediction Heads**:
  - Vulnerability (node-level)  
  - Adaptation (graph-level)  
  - Route shift (edge-level)  

---

## 📈 Outputs

All outputs are saved in `OUT_DIR`:

### 📊 Data
- `cleaned_agora.csv`

### 📉 Plots
- `loss_curves.png`
- `adaptation_over_time.png`
- `top_vulnerable_nodes.png`
- `top_route_shifts.png`
- `network_vulnerability.png`
- `temporal_vulnerability.png`
- `disruption_comparison.png`

### 🌐 Interactive Visualizations
- `interactive_network.html`
- `interactive_globe.html`

### 📄 Tables
- `vulnerability_scores_last_quarter.csv`
- `route_shift_scores_last_quarter.csv`
- `adaptation_scores_over_time.csv`

### 💾 Model
- `temporal_gnn_drug_network.pt`

---

## ⚙️ How to Run

```bash
python tdl_final.py
