# gnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# ✅ Define GNN Model
class GNNModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # ✅ Apply GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # ✅ Global pooling
        x = global_mean_pool(x, data.batch)

        # ✅ Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.view(-1)
