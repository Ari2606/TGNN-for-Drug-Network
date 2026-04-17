"""
drug_network_gnn.py
===================
Temporal Graph Neural Network for Dark-Web Drug Supply Network Analysis
(Agora Marketplace, 2014–2015)

Combines:
  • Data loading & cleaning
  • Quarterly feature extraction
  • Graph snapshot construction
  • TemporalMultiHeadGNN training (vulnerability / adaptation / route)
  • Static matplotlib plots
  • Interactive pyvis HTML network
  • NEW: Interactive Plotly Earth-style globe
  • Disruption-impact baseline comparison

Usage:
    python drug_network_gnn.py

Outputs (all saved to OUT_DIR):
    cleaned_agora.csv
    loss_curves.png
    adaptation_over_time.png
    top_vulnerable_nodes.png
    top_route_shifts.png
    vulnerability_scores_last_quarter.csv
    route_shift_scores_last_quarter.csv
    adaptation_scores_over_time.csv
    network_vulnerability.png
    temporal_vulnerability.png
    disruption_comparison.png
    interactive_network.html
    interactive_globe.html
    temporal_gnn_drug_network.pt
"""

import os
import sys
import math
import random
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import plotly.graph_objects as go

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, precision_score, recall_score,
)
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import to_networkx
from pyvis.network import Network

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================
AGORA_PATH   = "/Users/riasnair/Downloads/Agora.csv"
OUT_DIR      = "/Users/riasnair/Downloads/agora_temporal_gnn_outputs"
SEED         = 42
DEVICE       = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)
EPOCHS                  = 30
LR                      = 1e-3
DROPOUT                 = 0.25
HIDDEN                  = 128
MEM_DIM                 = 64
TOP_SELLERS_PER_QUARTER = 500
MIN_EDGE_WEIGHT         = 1e-8
TRAIN_QUARTERS          = ["2014Q1", "2014Q2", "2014Q3", "2014Q4", "2015Q1"]
TEST_QUARTERS           = ["2015Q2", "2015Q3"]
QUARTERS                = ["2014Q1", "2014Q2", "2014Q3", "2014Q4",
                           "2015Q1", "2015Q2", "2015Q3"]

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CATEGORY_MAP = {
    "cannabis": "Cannabis",  "weed": "Cannabis",        "hash": "Cannabis",
    "marijuana": "Cannabis", "concentrates": "Cannabis","edibles": "Cannabis",
    "seeds": "Cannabis",     "shake": "Cannabis",       "trim": "Cannabis",
    "synthetics": "Cannabis","cocaine": "Cocaine",      "crack": "Cocaine",
    "opioids": "Opioids",    "heroin": "Opioids",       "fentanyl": "Opioids",
    "morphine": "Opioids",   "codeine": "Opioids",      "hydrocodone": "Opioids",
    "oxycodone": "Opioids",  "buprenorphine": "Opioids","dihydrocodeine": "Opioids",
    "methadone": "Opioids",  "opium": "Opioids",        "benzos": "Opioids",
    "benzodiazepines": "Opioids",
    "ecstasy": "MDMA",       "mdma": "MDMA",            "mda": "MDMA",
    "molly": "MDMA",         "pills": "MDMA",
    "meth": "Meth",          "amphetamine": "Meth",     "ice": "Meth",
    "crystal": "Meth",       "speed": "Meth",
}
CATEGORY_ORDER = ["Cannabis", "Cocaine", "Opioids", "MDMA", "Meth"]


# ============================================================
# GEO MAPPING FOR GLOBE
# ============================================================
LOCATION_ALIASES = {
    "us": "usa",
    "u.s.": "usa",
    "u.s.a": "usa",
    "u.s.a.": "usa",
    "america": "usa",
    "united states of america": "usa",

    "uk": "united kingdom",
    "england": "united kingdom",
    "great britain": "united kingdom",

    "holland": "netherlands",

    "worldwide": "unknown",
    "world wide": "unknown",
    "international": "unknown",
    "eu": "unknown",
    "europe": "unknown",
    "domestic": "unknown",
    "stealth": "unknown",
    "stealth shipping": "unknown",
    "everywhere": "unknown",
}

LOCATION_COORDS = {
    "usa": (37.0902, -95.7129),
    "united states": (37.0902, -95.7129),
    "united kingdom": (55.3781, -3.4360),
    "canada": (56.1304, -106.3468),
    "australia": (-25.2744, 133.7751),
    "germany": (51.1657, 10.4515),
    "netherlands": (52.1326, 5.2913),
    "france": (46.2276, 2.2137),
    "spain": (40.4637, -3.7492),
    "india": (20.5937, 78.9629),
    "china": (35.8617, 104.1954),
    "italy": (41.8719, 12.5674),
    "belgium": (50.5039, 4.4699),
    "sweden": (60.1282, 18.6435),
    "norway": (60.4720, 8.4689),
    "denmark": (56.2639, 9.5018),
    "switzerland": (46.8182, 8.2275),
    "austria": (47.5162, 14.5501),
    "poland": (51.9194, 19.1451),
    "russia": (61.5240, 105.3188),
    "ukraine": (48.3794, 31.1656),
    "brazil": (-14.2350, -51.9253),
    "mexico": (23.6345, -102.5528),
    "unknown": (0.0, 0.0),
}


# ============================================================
# HELPERS
# ============================================================
def set_style():
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.grid"]      = True
    plt.rcParams["grid.alpha"]     = 0.25


def save_plot(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def normalize_category(raw):
    raw = str(raw).lower().strip()
    if not raw.startswith("drugs"):
        return None
    for key, val in CATEGORY_MAP.items():
        if key in raw:
            return val
    return None


def clean_price_series(series):
    s = series.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False)
    s = s.str.extract(r"([0-9]*\.?[0-9]+)", expand=False)
    return pd.to_numeric(s, errors="coerce")


def safe_rating_to_float(x):
    try:
        x = str(x).strip().replace(",", "")
        if x == "" or x.lower() == "nan":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def entropy_from_counter(counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = np.array(list(counter.values()), dtype=float) / total
    return float(-(probs * np.log(probs + 1e-12)).sum())


def quarter_sort_key(q):
    return (int(q[:4]), int(q[-1]))


def bce_from_probs(pred, true):
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    return float(-(true * np.log(pred) + (1 - true) * np.log(1 - pred)).mean())


def normalize_location(x):
    x = str(x).strip().lower()
    if x in LOCATION_ALIASES:
        x = LOCATION_ALIASES[x]
    if x == "" or x == "nan" or x == "none":
        return "unknown"
    return x


# ============================================================
# 1. LOAD + CLEAN
# ============================================================
def load_agora(path: str) -> pd.DataFrame:
    print(f"Loading: {path}")
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"Raw rows: {len(df):,}")

    df.columns = [
        c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        for c in df.columns
    ]

    rename_map = {
        "vendor": "seller", "category": "category", "item": "item",
        "item_description": "item_description", "price": "price",
        "origin": "ship_from", "destination": "ship_to",
        "rating": "rating", "remarks": "remarks",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    required_cols = ["seller", "category", "price", "ship_from", "ship_to"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["seller", "category", "ship_from", "ship_to",
                "item", "item_description", "remarks"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df.replace({"": np.nan, "nan": np.nan, "None": np.nan}, inplace=True)

    df["category_norm"] = df["category"].apply(normalize_category)
    df = df[df["category_norm"].notna()].copy()
    if len(df) == 0:
        raise ValueError("0 rows after drug filtering")

    df["price"] = clean_price_series(df["price"])
    df = df[df["price"].notna() & (df["price"] > 0)].copy()
    if len(df) == 0:
        raise ValueError("0 rows after price cleaning")

    if "rating" in df.columns:
        df["rating_num"] = df["rating"].apply(safe_rating_to_float)
    else:
        df["rating_num"] = np.nan

    for col in ["seller", "ship_from", "ship_to"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col] == "", col] = "Unknown"

    upper = df["price"].quantile(0.99)
    df = df[df["price"] <= upper].copy()

    np.random.seed(SEED)
    df["quarter"] = np.random.choice(QUARTERS, size=len(df))

    print(f"Clean rows: {len(df):,}")
    print(f"Unique sellers: {df['seller'].nunique():,}")
    print(f"Categories: {sorted(df['category_norm'].unique())}")
    return df.reset_index(drop=True)


# ============================================================
# 2. QUARTERLY FEATURE TABLES
# ============================================================
def build_quarterly_vendor_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    global_seller_activity = df.groupby("seller").size().to_dict()

    for q in sorted(df["quarter"].unique(), key=quarter_sort_key):
        qdf = df[df["quarter"] == q].copy()
        for seller, g in qdf.groupby("seller"):
            dest_counter = Counter(g["ship_to"].fillna("Unknown"))
            orig_counter = Counter(g["ship_from"].fillna("Unknown"))
            cat_counter  = Counter(g["category_norm"])
            prices  = g["price"].values
            ratings = (g["rating_num"].dropna().values
                       if "rating_num" in g.columns else np.array([]))

            row = {
                "quarter": q, "seller": seller,
                "listings":      len(g),
                "price_mean":    float(np.mean(prices))   if len(prices) else 0.0,
                "price_median":  float(np.median(prices)) if len(prices) else 0.0,
                "price_std":     float(np.std(prices))    if len(prices) else 0.0,
                "price_max":     float(np.max(prices))    if len(prices) else 0.0,
                "rating_mean":   float(np.mean(ratings))  if len(ratings) else 0.0,
                "rating_count":  float(len(ratings)),
                "unique_dest":   float(g["ship_to"].nunique()),
                "unique_orig":   float(g["ship_from"].nunique()),
                "cat_diversity": float(g["category_norm"].nunique()),
                "route_entropy": entropy_from_counter(dest_counter),
                "origin_entropy":entropy_from_counter(orig_counter),
                "activity_share":float(len(g) / len(qdf)),
                "seller_global_activity": float(global_seller_activity.get(seller, 0)),
                "quarter_index": float(QUARTERS.index(q)),
            }
            for cat in CATEGORY_ORDER:
                row[f"cat_{cat.lower()}"] = float(cat_counter.get(cat, 0))
            rows.append(row)

    vendor_df = pd.DataFrame(rows)
    vendor_df = vendor_df.sort_values(
        ["quarter", "listings"], ascending=[True, False]
    ).reset_index(drop=True)
    return vendor_df


def keep_top_sellers(vendor_df: pd.DataFrame,
                     top_n: int = TOP_SELLERS_PER_QUARTER) -> pd.DataFrame:
    kept = []
    for q, g in vendor_df.groupby("quarter"):
        kept.append(g.head(top_n))
    return pd.concat(kept, ignore_index=True)


# ============================================================
# 3. BUILD GRAPH SNAPSHOTS
# ============================================================
def build_graph_snapshots(df: pd.DataFrame):
    vendor_df = build_quarterly_vendor_table(df)
    vendor_df = keep_top_sellers(vendor_df, TOP_SELLERS_PER_QUARTER)

    feature_cols = [
        "listings", "price_mean", "price_median", "price_std", "price_max",
        "rating_mean", "rating_count", "unique_dest", "unique_orig",
        "cat_diversity", "route_entropy", "origin_entropy", "activity_share",
        "seller_global_activity", "quarter_index", "cat_cannabis",
    ]
    assert len(feature_cols) == 16

    train_vendor_df = vendor_df[vendor_df["quarter"].isin(TRAIN_QUARTERS)].copy()
    scaler_stats = {}
    for col in feature_cols:
        mu = train_vendor_df[col].mean()
        sd = train_vendor_df[col].std()
        scaler_stats[col] = (mu, sd if pd.notna(sd) and sd > 1e-8 else 1.0)

    snapshots      = []
    quarter_tables = {}

    for q in sorted(vendor_df["quarter"].unique(), key=quarter_sort_key):
        qtab    = vendor_df[vendor_df["quarter"] == q].copy().reset_index(drop=True)
        sellers = qtab["seller"].tolist()
        n2i     = {name: i for i, name in enumerate(sellers)}
        N       = len(sellers)

        x = qtab[feature_cols].copy()
        for col in feature_cols:
            mu, sd = scaler_stats[col]
            x[col] = (x[col] - mu) / sd
        x = torch.tensor(x.values, dtype=torch.float)

        qraw = df[(df["quarter"] == q) & (df["seller"].isin(sellers))].copy()
        edge_counter   = defaultdict(float)
        seller_to_idx  = n2i
        by_dest = qraw.groupby("ship_to")["seller"].apply(list)
        by_orig = qraw.groupby("ship_from")["seller"].apply(list)
        by_cat  = qraw.groupby("category_norm")["seller"].apply(list)

        def add_clique_edges(grouped, weight):
            for _, members in grouped.items():
                uniq = [m for m in set(members) if m in seller_to_idx]
                if len(uniq) < 2:
                    continue
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        u = seller_to_idx[uniq[i]]
                        v = seller_to_idx[uniq[j]]
                        edge_counter[(u, v)] += weight
                        edge_counter[(v, u)] += weight

        add_clique_edges(by_dest, 2.0)
        add_clique_edges(by_orig, 1.5)
        add_clique_edges(by_cat,  1.0)
        for i in range(N):
            edge_counter[(i, i)] += 1.0

        srcs, dsts, wts = [], [], []
        for (s, d), w in edge_counter.items():
            if w >= MIN_EDGE_WEIGHT:
                srcs.append(s)
                dsts.append(d)
                wts.append(math.log1p(w))
        if len(srcs) == 0:
            srcs, dsts, wts = [0], [0], [1.0]

        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_attr  = torch.tensor(wts, dtype=torch.float).unsqueeze(-1)

        quarter_tables[q] = qtab.copy()
        snapshots.append(Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            quarter=q, q_idx=QUARTERS.index(q),
            node_names=sellers, n2i=n2i,
        ))

    return snapshots, quarter_tables, feature_cols


# ============================================================
# 4. SELF-SUPERVISED LABELS
# ============================================================
def build_labels(snapshots, quarter_tables):
    vuln_labels, adapt_labels, route_labels = [], [], []
    quarter_list = [s.quarter for s in snapshots]

    for idx, snap in enumerate(snapshots):
        curr_q   = snap.quarter
        curr_tab = quarter_tables[curr_q].set_index("seller")
        if idx < len(snapshots) - 1:
            next_q   = quarter_list[idx + 1]
            next_tab = quarter_tables[next_q].set_index("seller")
        else:
            next_q, next_tab = curr_q, curr_tab

        v = torch.zeros(len(snap.node_names), dtype=torch.float)
        for i, seller in enumerate(snap.node_names):
            curr_l = float(curr_tab.loc[seller, "listings"]) if seller in curr_tab.index else 0.0
            next_l = float(next_tab.loc[seller, "listings"]) if seller in next_tab.index else 0.0
            if curr_l <= 0:
                score = 0.0
            else:
                drop   = max(0.0, curr_l - next_l) / max(curr_l, 1.0)
                vanish = 1.0 if seller not in next_tab.index else 0.0
                score  = min(1.0, 0.7 * drop + 0.3 * vanish)
            v[i] = score
        vuln_labels.append(v)

        cat_cols  = [c for c in curr_tab.columns if c.startswith("cat_")]
        curr_cat  = curr_tab[cat_cols].sum().values.astype(float)
        next_cat  = next_tab[cat_cols].sum().values.astype(float)
        curr_cat /= curr_cat.sum() + 1e-8
        next_cat /= next_cat.sum() + 1e-8
        adapt = np.abs(curr_cat - next_cat).mean()
        adapt_labels.append(torch.tensor([float(adapt)], dtype=torch.float))

        next_names = (set(snapshots[idx + 1].node_names)
                      if idx < len(snapshots) - 1 else set(snap.node_names))
        r = torch.zeros(snap.edge_index.size(1), dtype=torch.float)
        for e_idx, (s, d) in enumerate(snap.edge_index.t().tolist()):
            s_name = snap.node_names[s]
            d_name = snap.node_names[d]
            r[e_idx] = 1.0 if (s_name in next_names and d_name in next_names) else 0.0
        route_labels.append(r)

    return vuln_labels, adapt_labels, route_labels


# ============================================================
# 5. MODEL
# ============================================================
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim=16, hidden=128, out_dim=64, dropout=0.25):
        super().__init__()
        self.conv1  = SAGEConv(in_dim, hidden)
        self.conv2  = SAGEConv(hidden, hidden)
        self.conv3  = SAGEConv(hidden, out_dim)
        self.norm1  = nn.LayerNorm(hidden)
        self.norm2  = nn.LayerNorm(hidden)
        self.norm3  = nn.LayerNorm(out_dim)
        self.skip1  = nn.Linear(in_dim, hidden, bias=False)
        self.skip2  = nn.Linear(hidden, hidden, bias=False)
        self.skip3  = nn.Linear(hidden, out_dim, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h1 = self.drop(F.relu(self.norm1(self.conv1(x, edge_index) + self.skip1(x))))
        h2 = self.drop(F.relu(self.norm2(self.conv2(h1, edge_index) + self.skip2(h1))))
        h3 = self.drop(F.relu(self.norm3(self.conv3(h2, edge_index) + self.skip3(h2))))
        return h3


class TemporalMultiHeadGNN(nn.Module):
    def __init__(self, in_dim=16, hidden=128, memory_dim=64, dropout=0.25):
        super().__init__()
        self.encoder    = GraphSAGEEncoder(in_dim=in_dim, hidden=hidden,
                                           out_dim=memory_dim, dropout=dropout)
        self.gru        = nn.GRUCell(memory_dim, memory_dim)
        self.vuln_head  = nn.Sequential(
            nn.Linear(memory_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.adapt_head = nn.Sequential(
            nn.Linear(memory_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.route_head = nn.Sequential(
            nn.Linear(memory_dim * 2 + 1, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, snap, prev_memory):
        x          = snap.x
        edge_index = snap.edge_index
        edge_attr  = snap.edge_attr
        node_names = snap.node_names
        device     = x.device

        e_t    = self.encoder(x, edge_index)
        h_prev = torch.zeros(len(node_names), e_t.size(-1), device=device)
        for i, name in enumerate(node_names):
            if name in prev_memory:
                h_prev[i] = prev_memory[name]
        h_t = self.gru(e_t, h_prev)
        new_memory = {name: h_t[i].detach() for i, name in enumerate(node_names)}

        vuln  = self.vuln_head(h_t).squeeze(-1)
        batch = torch.zeros(h_t.size(0), dtype=torch.long, device=device)
        adapt = self.adapt_head(global_mean_pool(h_t, batch)).squeeze(-1)
        src, dst = edge_index[0], edge_index[1]
        route = self.route_head(
            torch.cat([h_t[src], h_t[dst], edge_attr], dim=-1)
        ).squeeze(-1)

        return vuln, adapt, route, new_memory


# ============================================================
# 6. TRAINING + EVALUATION
# ============================================================
def split_indices_by_quarter(snapshots):
    train_idx = [i for i, s in enumerate(snapshots) if s.quarter in TRAIN_QUARTERS]
    test_idx  = [i for i, s in enumerate(snapshots) if s.quarter in TEST_QUARTERS]
    return train_idx, test_idx


def train_model(model, snapshots, vuln_labels, adapt_labels, route_labels,
                epochs=30, lr=1e-3):
    model     = model.to(DEVICE)
    snapshots = [s.to(DEVICE) for s in snapshots]
    vuln_labels  = [v.to(DEVICE) for v in vuln_labels]
    adapt_labels = [a.to(DEVICE) for a in adapt_labels]
    route_labels = [r.to(DEVICE) for r in route_labels]

    train_idx, test_idx = split_indices_by_quarter(snapshots)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    history = {k: [] for k in
               ["train_total","train_vuln","train_adapt","train_route",
                "test_total", "test_vuln", "test_adapt", "test_route"]}

    for epoch in range(1, epochs + 1):
        model.train()
        mem_state   = {}
        totals      = {"total": 0.0, "vuln": 0.0, "adapt": 0.0, "route": 0.0}
        train_steps = 0

        for t, snap in enumerate(snapshots):
            if t not in train_idx:
                with torch.no_grad():
                    _, _, _, mem_state = model(snap, mem_state)
                continue
            opt.zero_grad()
            vp, ap, rp, mem_state = model(snap, mem_state)
            lv   = F.mse_loss(vp, vuln_labels[t])
            la   = F.mse_loss(ap, adapt_labels[t].view_as(ap))
            lr_  = F.binary_cross_entropy(rp.clamp(1e-6, 1 - 1e-6), route_labels[t])
            loss = lv + la + lr_
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            totals["total"] += loss.item()
            totals["vuln"]  += lv.item()
            totals["adapt"] += la.item()
            totals["route"] += lr_.item()
            train_steps += 1

        div = max(train_steps, 1)
        for k in ["total", "vuln", "adapt", "route"]:
            history[f"train_{k}"].append(totals[k] / div)

        test_m = evaluate_model(
            model, snapshots, vuln_labels, adapt_labels, route_labels,
            subset="test", print_metrics=False,
        )
        history["test_total"].append(test_m["total_loss"])
        history["test_vuln"].append(test_m["vulnerability_mse"])
        history["test_adapt"].append(test_m["adaptation_mse"])
        history["test_route"].append(test_m["route_bce"])

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train={history['train_total'][-1]:.4f} | "
                f"test={history['test_total'][-1]:.4f} | "
                f"route_f1={test_m['route_f1']:.4f}"
            )

    return model, history, snapshots, vuln_labels, adapt_labels, route_labels


@torch.no_grad()
def evaluate_model(model, snapshots, vuln_labels, adapt_labels, route_labels,
                   subset="test", print_metrics=True):
    model.eval()
    train_idx, test_idx = split_indices_by_quarter(snapshots)
    use_idx = (test_idx  if subset == "test"  else
               train_idx if subset == "train" else
               list(range(len(snapshots))))

    mem_state = {}
    vuln_true_all, vuln_pred_all   = [], []
    adapt_true_all, adapt_pred_all = [], []
    route_true_all, route_pred_all = [], []
    route_loss_all = []
    quarter_rows   = []

    for t, snap in enumerate(snapshots):
        vp, ap, rp, mem_state = model(snap, mem_state)
        if t not in use_idx:
            continue
        v_true = vuln_labels[t].detach().cpu().numpy()
        v_pred = vp.detach().cpu().numpy()
        a_true = adapt_labels[t].view_as(ap).detach().cpu().numpy()
        a_pred = ap.detach().cpu().numpy()
        r_true = route_labels[t].detach().cpu().numpy()
        r_pred = rp.detach().cpu().numpy()

        vuln_true_all.extend(v_true)
        vuln_pred_all.extend(v_pred)
        adapt_true_all.extend(a_true)
        adapt_pred_all.extend(a_pred)
        route_true_all.extend(r_true)
        route_pred_all.extend(r_pred)
        route_loss_all.append(bce_from_probs(r_pred, r_true))
        rb = (r_pred >= 0.5).astype(int)
        quarter_rows.append({
            "quarter":          snap.quarter,
            "num_nodes":        len(v_true),
            "num_edges":        len(r_true),
            "vulnerability_mse":float(mean_squared_error(v_true, v_pred)),
            "adaptation_mse":   float(mean_squared_error(a_true, a_pred)),
            "route_accuracy":   float(accuracy_score(r_true.astype(int), rb)),
            "route_f1":         float(f1_score(r_true.astype(int), rb, zero_division=0)),
        })

    vta = np.array(vuln_true_all)
    vpa = np.array(vuln_pred_all)
    ata = np.array(adapt_true_all)
    apa = np.array(adapt_pred_all)
    rta = np.array(route_true_all).astype(int)
    rpa = np.array(route_pred_all)
    rba = (rpa >= 0.5).astype(int)

    metrics = {
        "subset":             subset,
        "vulnerability_mse":  float(mean_squared_error(vta, vpa))  if len(vta) else 0.0,
        "vulnerability_mae":  float(mean_absolute_error(vta, vpa)) if len(vta) else 0.0,
        "adaptation_mse":     float(mean_squared_error(ata, apa))  if len(ata) else 0.0,
        "adaptation_mae":     float(mean_absolute_error(ata, apa)) if len(ata) else 0.0,
        "route_bce":          float(np.mean(route_loss_all))       if route_loss_all else 0.0,
        "route_accuracy":     float(accuracy_score(rta, rba))      if len(rta) else 0.0,
        "route_precision":    float(precision_score(rta, rba, zero_division=0)) if len(rta) else 0.0,
        "route_recall":       float(recall_score(rta, rba, zero_division=0))    if len(rta) else 0.0,
        "route_f1":           float(f1_score(rta, rba, zero_division=0))         if len(rta) else 0.0,
    }
    metrics["total_loss"] = (
        metrics["vulnerability_mse"]
        + metrics["adaptation_mse"]
        + metrics["route_bce"]
    )

    pd.DataFrame(quarter_rows).to_csv(
        os.path.join(OUT_DIR, f"{subset}_metrics_by_quarter.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUT_DIR, f"{subset}_metrics_summary.csv"), index=False)

    if print_metrics:
        print(f"\n=== {subset.upper()} EVALUATION ===")
        for k, v in metrics.items():
            if k != "subset":
                print(f"  {k:<22} {v:.4f}")

    return metrics


# ============================================================
# 7. INFERENCE
# ============================================================
@torch.no_grad()
def infer_all(model, snapshots):
    model.eval()
    mem_state = {}
    results   = []
    for snap in snapshots:
        vp, ap, rp, mem_state = model(snap, mem_state)
        results.append({
            "quarter":      snap.quarter,
            "node_names":   snap.node_names,
            "edge_index":   snap.edge_index.detach().cpu().numpy(),
            "vuln":         vp.detach().cpu().numpy(),
            "adapt":        float(ap.detach().cpu().item()),
            "route_scores": rp.detach().cpu().numpy(),
            "n2i":          snap.n2i,
        })
    return results


# ============================================================
# 8. STATIC PLOTS
# ============================================================
def plot_training(history):
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    plots = [
        ("train_total",  "test_total",  "Total Loss"),
        ("train_vuln",   "test_vuln",   "Vulnerability Loss"),
        ("train_adapt",  "test_adapt",  "Adaptation Loss"),
        ("train_route",  "test_route",  "Route Loss"),
    ]
    for ax, (tr, te, title) in zip(axes, plots):
        ax.plot(history[tr], marker="o", label="Train")
        ax.plot(history[te], marker="o", label="Test")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    save_plot("loss_curves.png")


def plot_adaptation(results):
    set_style()
    qs   = [r["quarter"] for r in results]
    vals = [r["adapt"]   for r in results]
    plt.figure(figsize=(10, 4))
    plt.plot(qs, vals, marker="o")
    plt.title("Predicted Graph Adaptation Over Time")
    plt.xlabel("Quarter")
    plt.ylabel("Adaptation Score")
    save_plot("adaptation_over_time.png")


def plot_top_vulnerable(results):
    set_style()
    last = results[-1]
    df_v = pd.DataFrame({"node": last["node_names"], "vuln": last["vuln"]})
    df_v = df_v.sort_values("vuln", ascending=False).head(20)
    plt.figure(figsize=(12, 7))
    plt.barh(df_v["node"][::-1], df_v["vuln"][::-1])
    plt.title(f"Top Vulnerable Vendors — {last['quarter']}")
    plt.xlabel("Predicted Vulnerability")
    save_plot("top_vulnerable_nodes.png")


def plot_top_route_shifts(results):
    set_style()
    last  = results[-1]
    ei    = last["edge_index"]
    names = last["node_names"]
    pairs = sorted(
        [(names[ei[0, i]], names[ei[1, i]], float(last["route_scores"][i]))
         for i in range(len(last["route_scores"]))],
        key=lambda x: -x[2]
    )[:20]
    labels = [f"{s[:18]} -> {d[:18]}" for s, d, _ in pairs]
    vals   = [v for _, _, v in pairs]
    plt.figure(figsize=(13, 7))
    plt.barh(labels[::-1], vals[::-1])
    plt.title(f"Top Predicted Route Shifts — {last['quarter']}")
    plt.xlabel("Route Shift Score")
    save_plot("top_route_shifts.png")


def save_results_tables(results):
    last = results[-1]
    ei   = last["edge_index"]

    pd.DataFrame({"seller": last["node_names"], "vulnerability": last["vuln"]}) \
      .sort_values("vulnerability", ascending=False) \
      .to_csv(os.path.join(OUT_DIR, "vulnerability_scores_last_quarter.csv"), index=False)

    pd.DataFrame({
        "src": [last["node_names"][ei[0, i]] for i in range(ei.shape[1])],
        "dst": [last["node_names"][ei[1, i]] for i in range(ei.shape[1])],
        "route_shift_score": last["route_scores"],
    }).sort_values("route_shift_score", ascending=False) \
      .to_csv(os.path.join(OUT_DIR, "route_shift_scores_last_quarter.csv"), index=False)

    pd.DataFrame({
        "quarter": [r["quarter"] for r in results],
        "adaptation_score": [r["adapt"] for r in results],
    }).to_csv(os.path.join(OUT_DIR, "adaptation_scores_over_time.csv"), index=False)


# ============================================================
# 9. INTERACTIVE VISUALISATIONS
# ============================================================
def _get_vuln_scores_for_snap(model, snapshots, target_idx: int):
    model.eval()
    mem_state = {}
    with torch.no_grad():
        for t, snap in enumerate(snapshots):
            vp, ap, rp, mem_state = model(snap, mem_state)
            if t == target_idx:
                return torch.nan_to_num(vp, nan=0.5).clamp(0, 1).cpu().numpy()
    return np.array([])


def plot_network_static(snapshot, vuln_scores, save_path, top_k=15,
                        max_nodes=100, max_edges=500):
    print("[Viz] Building static network plot …")
    edge_index = snapshot.edge_index.cpu().numpy()
    node_names = snapshot.node_names

    keep = set(np.argsort(vuln_scores)[-max_nodes:].tolist())
    print(f"[Viz] Keeping top-{max_nodes} nodes by vulnerability …")

    G = nx.DiGraph()
    edge_count = 0
    for s, d in edge_index.T.tolist():
        if s in keep and d in keep and s != d:
            G.add_edge(s, d)
            edge_count += 1
            if edge_count >= max_edges:
                break
    print(f"[Viz] Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

    if len(G.nodes()) == 0:
        print("[Viz] Nothing to draw.")
        return

    node_list = list(G.nodes())
    pos = nx.spring_layout(G.to_undirected(), seed=42, k=2.0, iterations=30)

    scores = np.array([vuln_scores[n] if n < len(vuln_scores) else 0.5
                       for n in node_list])
    norm   = plt.Normalize(vmin=0, vmax=1)
    cmap   = plt.cm.RdYlBu_r
    top_k_set = set(np.argsort(vuln_scores)[-top_k:].tolist()) & set(node_list)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#080e1a")
    ax.set_facecolor("#080e1a")

    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#334155", alpha=0.3, width=0.4,
        arrows=True, arrowsize=6, connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=node_list,
        node_color=[cmap(norm(s)) for s in scores],
        node_size=[40 + 160 * (vuln_scores[n] if n < len(vuln_scores) else 0.5)
                   for n in node_list],
        alpha=0.8, ax=ax,
    )
    if top_k_set:
        critical = list(top_k_set)
        nx.draw_networkx_nodes(
            G, pos, nodelist=critical,
            node_color="none", edgecolors="#ff4444",
            node_size=500, linewidths=2.5, ax=ax,
        )
        labels = {n: node_names[n][:15] for n in critical[:8] if n < len(node_names)}
        nx.draw_networkx_labels(
            G, pos, labels=labels,
            font_color="white", font_size=6.5, font_weight="bold", ax=ax,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
    cbar.set_label("Vulnerability Score", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.legend(
        handles=[
            mpatches.Patch(facecolor=cmap(0.9), label="High Vulnerability"),
            mpatches.Patch(facecolor=cmap(0.1), label="Low Vulnerability"),
            mpatches.Patch(facecolor="none", edgecolor="#ff4444",
                           linewidth=2, label=f"Top-{top_k} Critical"),
        ],
        loc="upper left", facecolor="#0f172a", edgecolor="#334155",
        labelcolor="white", fontsize=9, framealpha=0.9,
    )
    ax.set_title(
        f"Agora Drug Supply Network — Vulnerability Map\n"
        f"{len(node_list)} nodes | {len(G.edges())} edges | "
        f"Red ring = structurally critical",
        color="white", fontsize=13, fontweight="bold", pad=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Viz] Static network saved → {save_path}")


def plot_temporal_vulnerability(model, snapshots, save_path):
    labels, means = [], []
    model.eval()
    mem_state = {}
    with torch.no_grad():
        for snap in snapshots:
            vp, _, _, mem_state = model(snap, mem_state)
            vp = torch.nan_to_num(vp, nan=0.5).clamp(0, 1)
            means.append(vp.mean().item())
            labels.append(snap.quarter)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(labels, means, marker="o", color="#0D9488", linewidth=2.5, markersize=8)
    ax.fill_between(labels, means, alpha=0.15, color="#0D9488")
    for x, y in zip(labels, means):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="#0D9488")
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel("Mean Vulnerability Score", fontsize=11)
    ax.set_title("Network Vulnerability Over Time — Agora 2014–2015",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Temporal vulnerability saved → {save_path}")


def plot_disruption_comparison(snapshot, vuln_scores, save_path):
    print("[Viz] Computing disruption impact (may take ~30 s) …")
    G  = to_networkx(snapshot, to_undirected=True)
    N  = snapshot.num_nodes
    deg = torch.zeros(N)
    for s in snapshot.edge_index[0]:
        deg[s] += 1
    degree_scores = deg.numpy()

    def efficiency_drop(G_base, remove_idx):
        G2 = G_base.copy()
        G2.remove_nodes_from([i for i in remove_idx if i in G2])
        if len(G2) < 2:
            return 1.0
        sample = list(G_base.nodes())[:50]
        base_eff = sum(
            1 / v
            for src in sample
            for v in nx.single_source_shortest_path_length(G_base, src).values()
            if v > 0
        )
        new_eff = sum(
            1 / v
            for src in list(G2.nodes())[:50]
            for v in nx.single_source_shortest_path_length(G2, src).values()
            if v > 0
        )
        return max((base_eff - new_eff) / (base_eff + 1e-9), 0)

    k_vals = [5, 10, 20, 50]
    model_drops, degree_drops, random_drops = [], [], []
    for k in k_vals:
        k = min(k, N - 2)
        model_drops.append(efficiency_drop(G, np.argsort(vuln_scores)[-k:].tolist()) * 100)
        degree_drops.append(efficiency_drop(G, np.argsort(degree_scores)[-k:].tolist()) * 100)
        random_drops.append(np.mean([
            efficiency_drop(G, np.random.choice(N, k, replace=False).tolist()) * 100
            for _ in range(5)
        ]))

    x = np.arange(len(k_vals))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, model_drops,  w, label="TGNN (Ours)",       color="#EF4444", alpha=0.9)
    ax.bar(x,     degree_drops, w, label="Degree Centrality", color="#3B82F6", alpha=0.9)
    ax.bar(x + w, random_drops, w, label="Random Removal",    color="#94A3B8", alpha=0.9)
    ax.set_xlabel("Nodes Removed (k)", fontsize=11)
    ax.set_ylabel("Network Efficiency Drop (%)", fontsize=11)
    ax.set_title("Disruption Impact: TGNN vs Baselines — Agora Network",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Disruption comparison saved → {save_path}")


def plot_network_interactive(snapshot, vuln_scores, save_path,
                             max_nodes=80, max_edges=200):
    print(f"[Viz] Building interactive HTML graph (top {max_nodes} nodes, {max_edges} edges) …")
    edge_index = snapshot.edge_index.cpu().numpy()
    node_names = snapshot.node_names

    keep_idx = set(np.argsort(vuln_scores)[-max_nodes:].tolist())

    scored_edges = []
    for e in range(edge_index.shape[1]):
        s, d = int(edge_index[0, e]), int(edge_index[1, e])
        if s == d or s not in keep_idx or d not in keep_idx:
            continue
        scored_edges.append((s, d, vuln_scores[s] + vuln_scores[d]))
    scored_edges.sort(key=lambda x: -x[2])
    top_edges = scored_edges[:max_edges]

    edge_nodes = set()
    for s, d, _ in top_edges:
        edge_nodes.add(s)
        edge_nodes.add(d)

    net = Network(
        height="820px", width="100%",
        bgcolor="#080e1a", font_color="white",
        directed=True,
    )
    net.set_options("""
    {
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.06,
          "damping": 0.5
        },
        "minVelocity": 0.5,
        "maxVelocity": 30,
        "stabilization": { "iterations": 150 }
      },
      "interaction": { "tooltipDelay": 80, "hideEdgesOnDrag": true }
    }
    """)

    for i in edge_nodes:
        score = float(vuln_scores[i]) if i < len(vuln_scores) else 0.5
        name  = node_names[i] if i < len(node_names) else f"Node {i}"
        color = ("#ef4444" if score > 0.7 else
                 "#f59e0b" if score > 0.4 else
                 "#3b82f6")
        net.add_node(
            i,
            label=name[:20],
            title=(
                f"<div style='font-family:monospace;font-size:12px;padding:4px'>"
                f"<b>{name}</b><br>"
                f"Vulnerability: <b style='color:{color}'>{score:.3f}</b>"
                f"</div>"
            ),
            color={"background": color, "border": "#ffffff",
                   "highlight": {"border": "#ffffff", "background": color}},
            size=8 + score * 28,
            borderWidth=1,
        )

    for s, d, _ in top_edges:
        net.add_edge(
            s, d,
            title=f"<span style='font-size:11px'>{node_names[s]} → {node_names[d]}</span>",
            color={"color": "#475569", "opacity": 0.7},
            width=1.0,
            arrows="to",
        )

    net.write_html(save_path)
    print(f"[Viz] Interactive HTML saved → {save_path} ({len(edge_nodes)} nodes, {len(top_edges)} edges)")


# ============================================================
# 10. GLOBE VISUALIZATION
# ============================================================
def build_geo_route_table(df):
    g = df.copy()

    g["ship_from_norm"] = g["ship_from"].apply(normalize_location)
    g["ship_to_norm"]   = g["ship_to"].apply(normalize_location)

    g = g[
        g["ship_from_norm"].isin(LOCATION_COORDS.keys()) &
        g["ship_to_norm"].isin(LOCATION_COORDS.keys())
    ].copy()

    if len(g) == 0:
        return pd.DataFrame(), pd.DataFrame()

    routes = (
        g.groupby(["ship_from_norm", "ship_to_norm"])
         .size()
         .reset_index(name="weight")
         .sort_values("weight", ascending=False)
    )

    node_from = (
        g.groupby("ship_from_norm")
         .size()
         .reset_index(name="activity")
         .rename(columns={"ship_from_norm": "location"})
    )

    node_to = (
        g.groupby("ship_to_norm")
         .size()
         .reset_index(name="activity")
         .rename(columns={"ship_to_norm": "location"})
    )

    nodes = pd.concat([node_from, node_to], ignore_index=True)
    nodes = nodes.groupby("location", as_index=False)["activity"].sum()

    nodes["lat"] = nodes["location"].map(lambda x: LOCATION_COORDS[x][0])
    nodes["lon"] = nodes["location"].map(lambda x: LOCATION_COORDS[x][1])

    routes["from_lat"] = routes["ship_from_norm"].map(lambda x: LOCATION_COORDS[x][0])
    routes["from_lon"] = routes["ship_from_norm"].map(lambda x: LOCATION_COORDS[x][1])
    routes["to_lat"]   = routes["ship_to_norm"].map(lambda x: LOCATION_COORDS[x][0])
    routes["to_lon"]   = routes["ship_to_norm"].map(lambda x: LOCATION_COORDS[x][1])

    return nodes, routes


def plot_interactive_globe(df, save_path, top_n_routes=100):
    print("[Globe] Building interactive globe …")
    nodes, routes = build_geo_route_table(df)

    if len(nodes) == 0 or len(routes) == 0:
        print("[Globe] No valid geo routes found.")
        return

    routes = routes.head(top_n_routes)

    fig = go.Figure()

    for _, row in routes.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[row["from_lon"], row["to_lon"]],
            lat=[row["from_lat"], row["to_lat"]],
            mode="lines",
            line=dict(
                width=max(1, min(6, row["weight"] / 5)),
                color="rgba(255,165,0,0.45)"
            ),
            opacity=0.75,
            hoverinfo="text",
            text=f'{row["ship_from_norm"]} → {row["ship_to_norm"]}<br>Weight: {row["weight"]}'
        ))

    fig.add_trace(go.Scattergeo(
        lon=nodes["lon"],
        lat=nodes["lat"],
        mode="markers+text",
        text=nodes["location"],
        textposition="top center",
        marker=dict(
            size=np.clip(nodes["activity"] * 1.2, 8, 30),
            color=nodes["activity"],
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="Activity"),
            line=dict(width=0.8, color="white"),
            opacity=0.92
        ),
        hovertext=[
            f"{loc}<br>Activity: {act}"
            for loc, act in zip(nodes["location"], nodes["activity"])
        ],
        hoverinfo="text"
    ))

    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        landcolor="rgb(28,35,52)",
        showocean=True,
        oceancolor="rgb(5,10,25)",
        showlakes=True,
        lakecolor="rgb(5,10,25)",
        showcountries=True,
        countrycolor="rgba(220,220,220,0.2)",
        coastlinecolor="rgba(255,255,255,0.25)",
        showcoastlines=True,
        bgcolor="rgb(3,6,18)"
    )

    fig.update_layout(
        title="Interactive Global Drug Route Network",
        paper_bgcolor="rgb(3,6,18)",
        plot_bgcolor="rgb(3,6,18)",
        font=dict(color="white"),
        height=850,
        margin=dict(l=0, r=0, t=55, b=0)
    )

    fig.write_html(save_path)
    print(f"[Globe] Interactive globe saved → {save_path}")


# ============================================================
# 11. MAIN
# ============================================================
def main():
    df = load_agora(AGORA_PATH)
    cleaned_path = os.path.join(OUT_DIR, "cleaned_agora.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"Saved: {cleaned_path}")

    snapshots, quarter_tables, feature_cols = build_graph_snapshots(df)
    print(f"Built {len(snapshots)} graph snapshots:")
    for snap in snapshots:
        print(f"  {snap.quarter}: nodes={snap.x.size(0)}, edges={snap.edge_index.size(1)}")

    vuln_labels, adapt_labels, route_labels = build_labels(snapshots, quarter_tables)

    model = TemporalMultiHeadGNN(
        in_dim=len(feature_cols), hidden=HIDDEN,
        memory_dim=MEM_DIM, dropout=DROPOUT,
    )

    model, history, snapshots, vuln_labels, adapt_labels, route_labels = train_model(
        model, snapshots, vuln_labels, adapt_labels, route_labels,
        epochs=EPOCHS, lr=LR,
    )

    train_metrics = evaluate_model(model, snapshots, vuln_labels, adapt_labels,
                                   route_labels, subset="train")
    test_metrics  = evaluate_model(model, snapshots, vuln_labels, adapt_labels,
                                   route_labels, subset="test")
    all_metrics   = evaluate_model(model, snapshots, vuln_labels, adapt_labels,
                                   route_labels, subset="all")

    results = infer_all(model, snapshots)

    plot_training(history)
    plot_adaptation(results)
    plot_top_vulnerable(results)
    plot_top_route_shifts(results)
    save_results_tables(results)

    last_idx    = len(snapshots) - 1
    vuln_scores = _get_vuln_scores_for_snap(model, snapshots, last_idx)
    last_snap   = snapshots[last_idx]

    plot_network_static(
        last_snap, vuln_scores,
        save_path=os.path.join(OUT_DIR, "network_vulnerability.png"),
        top_k=15, max_nodes=100, max_edges=500,
    )

    plot_temporal_vulnerability(
        model, snapshots,
        save_path=os.path.join(OUT_DIR, "temporal_vulnerability.png"),
    )

    plot_disruption_comparison(
        last_snap, vuln_scores,
        save_path=os.path.join(OUT_DIR, "disruption_comparison.png"),
    )

    plot_network_interactive(
        last_snap, vuln_scores,
        save_path=os.path.join(OUT_DIR, "interactive_network.html"),
        max_nodes=80, max_edges=200,
    )

    # NEW GLOBE
    plot_interactive_globe(
        df,
        save_path=os.path.join(OUT_DIR, "interactive_globe.html"),
        top_n_routes=100
    )

    ckpt_path = os.path.join(OUT_DIR, "temporal_gnn_drug_network.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "all_metrics": all_metrics,
            "results": results,
            "feature_dim": len(feature_cols),
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint: {ckpt_path}")

    print("\n" + "=" * 55)
    print("All outputs written to:", OUT_DIR)
    print("  cleaned_agora.csv")
    print("  loss_curves.png")
    print("  adaptation_over_time.png")
    print("  top_vulnerable_nodes.png")
    print("  top_route_shifts.png")
    print("  network_vulnerability.png")
    print("  temporal_vulnerability.png")
    print("  disruption_comparison.png")
    print("  interactive_network.html")
    print("  interactive_globe.html       ← open in browser")
    print("  temporal_gnn_drug_network.pt")
    print("=" * 55)


if __name__ == "__main__":
    main()