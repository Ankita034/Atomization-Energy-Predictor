from flask import Flask, render_template, request
import numpy as np
import joblib
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ✅ Load Models
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ✅ Load High Correlation Indices (Feature Selection Indices)
with open("high_indices.pkl", "rb") as f:
    high_correlation_indices = pickle.load(f)

# ✅ Dummy Placeholder for T_scaled and Median Calculation
T_scaled = np.zeros((7165,))  # Dummy placeholder (replace if real values are available)
T_median = np.median(scaler.inverse_transform(T_scaled.reshape(-1, 1)))

# ✅ Load GNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GNNModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1)


# ✅ Load GNN Model State
gnn_model = GNNModel(hidden_dim=64).to(device)
gnn_model.load_state_dict(torch.load("gnn_model.pth"))
gnn_model.eval()

# ✅ Load Graph Data
with open("graph_data.pkl", "rb") as f:
    graph_data = pickle.load(f)


# ✅ Home Page
@app.route("/")
def home():
    return render_template("home.html")


# ✅ Input Page
@app.route("/input")
def input_page():
    return render_template("input.html")


# ✅ Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    molecule_index = int(request.form["molecule_id"])
    selected_model = request.form["model"]

    # ✅ Load Graph Data for Molecule
    data = graph_data[molecule_index].to(device)

    # ✅ Get True Energy
    true_energy = scaler.inverse_transform([[data.y.item()]])[0, 0]

    # ✅ Model Predictions
    if selected_model == "gnn":
        with torch.no_grad():
            predicted_energy = gnn_model(data).item()

    else:
        # ✅ Flatten molecule data for XGB and RF
        X_molecule = data.x.cpu().numpy().flatten().reshape(1, -1)

        # ✅ Convert to Coulomb Matrix
        coulomb_matrix = np.zeros((23, 23))
        np.fill_diagonal(coulomb_matrix, X_molecule.flatten())

        # ✅ Flatten Coulomb matrix
        X_molecule_flat = coulomb_matrix.flatten().reshape(1, -1)

        # ✅ Apply Feature Selection to Match Model
        if X_molecule_flat.shape[1] > max(high_correlation_indices):
            X_molecule_selected = np.delete(X_molecule_flat, list(high_correlation_indices), axis=1)
        else:
            X_molecule_selected = X_molecule_flat

        # ✅ Correct Model Prediction
        if selected_model == "xgb":
            predicted_energy = xgb_model.predict(X_molecule_selected)[0]
        elif selected_model == "rf":
            predicted_energy = rf_model.predict(X_molecule_selected)[0]

    # ✅ Inverse Scale Prediction
    predicted_energy = scaler.inverse_transform([[predicted_energy]])[0, 0]

    # ✅ Determine Reactivity
    def infer_reactivity(energy, threshold):
        return "Reactive ⚡" if energy >= threshold else "Stable 🧪"

    reactivity_result = infer_reactivity(predicted_energy, T_median)

    return render_template(
        "prediction.html",
        molecule_index=molecule_index,
        true_energy=true_energy,
        predicted_energy=predicted_energy,
        reactivity_result=reactivity_result,
        selected_model=selected_model.upper(),
    )


if __name__ == "__main__":
    app.run(debug=True)
