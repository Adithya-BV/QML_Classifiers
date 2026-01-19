import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import os

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
n_qubits = 4
n_layers = 2
batch_size = 32
epochs = 2  # Keep small for simulation time
learning_rate = 0.01

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

# --- Quantum Circuit ---
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
except:
    print("Lightning qubit not found, using default.qubit")
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
        # 4 qubits measured -> 4 inputs to linear layer
        self.fc = nn.Linear(n_qubits, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.q_layer(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def train_model():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = HybridModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    print("\nStarting Hybrid QML Training...", flush=True)
    print(f"Device: {dev.name}, Qubits: {n_qubits}, Layers: {n_layers}", flush=True)
    print(f"Train Size: {len(X_train)}, Batch Size: {batch_size}, Epochs: {epochs}", flush=True)
    
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
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)
                
        avg_loss = total_loss / batches
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}", flush=True)
        
    print(f"Training Time: {time.time() - start_time:.2f}s")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_loss = 0
        all_preds = []
        all_targets = []
        
        # Test on full test set (careful with memory/time if huge, but 20k samples with forward pass is okay)
        # Batching test set to be safe
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        
        print("\nEvaluating on Test Set...")
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            all_preds.append(output.numpy())
            all_targets.append(target.numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_targets, all_preds > 0.5)
    auc = roc_auc_score(all_targets, all_preds)
    
    print("\n--- QML Model Results ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Save Results for Comparison (Optional)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    np.save(os.path.join(data_dir, 'qml_preds.npy'), all_preds)
    np.save(os.path.join(data_dir, 'qml_targets.npy'), all_targets)

if __name__ == "__main__":
    train_model()
