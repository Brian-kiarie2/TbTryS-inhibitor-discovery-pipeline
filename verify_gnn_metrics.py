"""
VERIFY GNN METRICS FROM SAVED CHECKPOINT
=========================================
This script loads the trained checkpoint and computes all reported metrics
on the held-out validation set, providing a fully traceable record.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

# ============================================
# 1. REPRODUCE THE EXACT MODEL ARCHITECTURE
# ============================================
class GCN(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64):
        super().__init__()
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.out = Linear(embedding_size * 2, 1)

    def forward(self, x, edge_index, batch_index):
        x = x.float()
        hidden = torch.tanh(self.initial_conv(x, edge_index))
        hidden = torch.tanh(self.conv1(hidden, edge_index))
        hidden = torch.tanh(self.conv2(hidden, edge_index))
        hidden = torch.tanh(self.conv3(hidden, edge_index))
        hidden = torch.cat([global_max_pool(hidden, batch_index),
                            global_mean_pool(hidden, batch_index)], dim=1)
        return self.out(hidden)

# ============================================
# 2. LOAD DATASET WITH FIXED SEED
# ============================================
device = torch.device('cpu')

# CRITICAL: Set seed BEFORE shuffle to reproduce the split
torch.manual_seed(42)
np.random.seed(42)

dataset = MoleculeNet(root="data", name="Lipo")
print(f"Total dataset size: {len(dataset)}")
print(f"Node features: {dataset.num_node_features}")

# Reproduce the 80/20 split
dataset = dataset.shuffle()
split_idx = int(0.8 * len(dataset))
val_dataset = dataset[split_idx:]

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(f"Validation set size: {len(val_dataset)}")

# ============================================
# 3. LOAD THE CHECKPOINT
# ============================================
model = GCN(num_features=dataset.num_node_features).to(device)
state_dict = torch.load("gnn_trained_9features.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("✅ Checkpoint loaded successfully")

# ============================================
# 4. RUN INFERENCE ON VALIDATION SET
# ============================================
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        all_preds.extend(out.cpu().numpy().flatten())
        all_targets.extend(batch.y.cpu().numpy().flatten())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# ============================================
# 5. COMPUTE ALL METRICS
# ============================================
r2 = r2_score(all_targets, all_preds)
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
mae = mean_absolute_error(all_targets, all_preds)

pearson_r, pearson_p = stats.pearsonr(all_targets, all_preds)
spearman_r, spearman_p = stats.spearmanr(all_targets, all_preds)

# ============================================
# 6. PRINT VERIFIABLE RESULTS
# ============================================
print("\n" + "=" * 60)
print("GNN PERFORMANCE ON HELD-OUT VALIDATION SET")
print("=" * 60)
print(f"Validation set size (n):  {len(all_targets)}")
print(f"R²:                       {r2:.4f}")
print(f"RMSE:                     {rmse:.4f}")
print(f"MAE:                      {mae:.4f}")
print(f"Pearson r:                {pearson_r:.4f}  (p = {pearson_p:.4e})")
print(f"Spearman ρ:               {spearman_r:.4f}  (p = {spearman_p:.4e})")
print("=" * 60)

# ============================================
# 7. SAVE TO FILE FOR THESIS RECORD
# ============================================
with open("gnn_validation_metrics.txt", "w") as f:
    f.write("GNN VALIDATION METRICS - VERIFIED FROM CHECKPOINT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Checkpoint: gnn_trained_9features.pt\n")
    f.write(f"Validation set size (n): {len(all_targets)}\n")
    f.write(f"R2: {r2:.6f}\n")
    f.write(f"RMSE: {rmse:.6f}\n")
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"Pearson_r: {pearson_r:.6f}\n")
    f.write(f"Pearson_p: {pearson_p:.6e}\n")
    f.write(f"Spearman_r: {spearman_r:.6f}\n")
    f.write(f"Spearman_p: {spearman_p:.6e}\n")

np.save("gnn_val_predictions.npy", all_preds)
np.save("gnn_val_targets.npy", all_targets)

print("\n✅ Metrics saved to: gnn_validation_metrics.txt")
print("✅ Predictions saved to: gnn_val_predictions.npy")
print("✅ Targets saved to: gnn_val_targets.npy")
