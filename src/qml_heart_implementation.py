import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
import time

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
n_qubits = 4
n_layers = 2
batch_size = 16 # Smaller batch for smaller dataset
epochs = 10 # More epochs since dataset is small (1000 samples)
learning_rate = 0.05

import os

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    X_train =  np.load(os.path.join(data_dir, 'X_train_heart.npy'))
    X_test =   np.load(os.path.join(data_dir, 'X_test_heart.npy'))
    y_train =  np.load(os.path.join(data_dir, 'y_train_heart.npy'))
    y_test =   np.load(os.path.join(data_dir, 'y_test_heart.npy'))
    
    # Convert to Torch Tensors
    tensor_x_train = torch.tensor(X_train, dtype=torch.float32)
    tensor_y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    tensor_x_test = torch.tensor(X_test, dtype=torch.float32)
    tensor_y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    return tensor_x_train, tensor_y_train, tensor_x_test, tensor_y_test

# --- Quantum Circuit ---
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
except:
    dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# --- Hybrid Model ---
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc = nn.Linear(n_qubits, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.q_layer(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def train_model():
    print("Loading Heart Disease data...")
    X_train, y_train, X_test, y_test = load_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = HybridModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    print("\nStarting QML Training (Heart Disease)...", flush=True)
    print(f"Train Size: {len(X_train)}, Epochs: {epochs}", flush=True)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}", flush=True)
        
    print(f"Training Time: {time.time() - start_time:.2f}s")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).numpy()
    
    accuracy = accuracy_score(y_test, test_preds > 0.5)
    auc = roc_auc_score(y_test, test_preds)
    
    print("\n--- QML Model Results (Heart Disease) ---")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")

if __name__ == "__main__":
    train_model()
