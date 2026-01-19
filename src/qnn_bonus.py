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
n_layers = 4  # More layers for data re-uploading to be effective
batch_size = 32
epochs = 2 
learning_rate = 0.01

import os

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    X_train =  np.load(os.path.join(data_dir, 'X_train_fraud_detection.npy'))
    X_test =   np.load(os.path.join(data_dir, 'X_test_fraud_detection.npy'))
    y_train =  np.load(os.path.join(data_dir, 'y_train_fraud_detection.npy'))
    y_test =   np.load(os.path.join(data_dir, 'y_test_fraud_detection.npy'))
    
    # Convert to Torch Tensors
    tensor_x_train = torch.tensor(X_train, dtype=torch.float32)
    tensor_y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    tensor_x_test = torch.tensor(X_test, dtype=torch.float32)
    tensor_y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    return tensor_x_train, tensor_y_train, tensor_x_test, tensor_y_test

# --- Quantum Circuit (Data Re-uploading) ---
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
except:
    dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    # weights shape: (n_layers, n_qubits, 3)
    
    for l in range(n_layers):
        # Data Re-uploading: Re-encode inputs in every layer
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y') # Rotate around Y
        
        # Variational Layer
        qml.StronglyEntanglingLayers(weights[l:l+1], wires=range(n_qubits))
        
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# --- QNN Model ---
class QNNModel(nn.Module):
    def __init__(self):
        super(QNNModel, self).__init__()
        # weights: n_layers blocks of StrongEntangling (each has 1 layer within the func call context per loop)
        # Actually StronglyEntanglingLayers takes weights of shape (L, M, 3). 
        # Here we slice weights so each re-uploading block gets 1 layer of entanglement.
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        
        # Post-processing: 4 measurements -> 1 output
        # To make it a "pure" QNN (or close to), we can just sum expectation values or use a small fixed weight
        # But for comparable classification, a final linear layer is standard interpretation.
        # Alternatively, we can project to parity. Let's stick to the hybrid head for consistency in optimization.
        self.fc = nn.Linear(n_qubits, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.q_layer(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def train_model():
    print("Loading Fraud data for Bonus QNN...")
    X_train, y_train, X_test, y_test = load_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = QNNModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    print("\nStarting Bonus QNN Training (Data Re-uploading)...", flush=True)
    print(f"Device: {dev.name}, Qubits: {n_qubits}, Layers (Re-uploads): {n_layers}", flush=True)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batches = 0 
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            if batch_idx % 100 == 0:
                 print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss {loss.item():.4f}", flush=True)
            
        avg_loss = total_loss / batches
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}", flush=True)
        
    print(f"Training Time: {time.time() - start_time:.2f}s")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Batch evaluation to avoid OOM or timeouts
        test_preds = []
        test_loader = DataLoader(TensorDataset(X_test), batch_size=batch_size)
        for (x_batch,) in test_loader:
             test_preds.append(model(x_batch))
        test_preds = torch.cat(test_preds).numpy()
    
    accuracy = accuracy_score(y_test, test_preds > 0.5)
    auc = roc_auc_score(y_test, test_preds)
    
    print("\n--- Bonus QNN Results ---")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")

if __name__ == "__main__":
    train_model()
