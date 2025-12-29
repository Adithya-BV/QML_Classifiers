# QML-Classifier for Credit Card Fraud Detection

## Project Overview
This project focuses on designing and benchmarking a Quantum Machine Learning (QML) classifier capable of handling noisy, non-convex classification tasks. Specifically, it addresses financial fraud detection by mapping classical transaction data into a high-dimensional quantum Hilbert space using a Variational Quantum Classifier (VQC).

## Phase 1: Data Pre-processing & Classical Benchmarking
### Data Strategy
- **Feature Selection:** Due to 4-qubit NISQ hardware constraints, we reduced the 8-feature dataset to the top 4 features using Random Forest importance: `ratio_to_median_purchase_price`, `online_order`, `distance_from_home`, and `used_pin_number`.
- **Cleaning:** Applied 99th percentile capping to numerical outliers to ensure stability during quantum state encoding.
- **Normalization:** Scaled features to $[0, 1]$ to accommodate the periodic nature of quantum rotation gates.

### Classical Performance Table
| Model | Accuracy | AUC-ROC |
| :--- | :--- | :--- |
| Logistic Regression | 95.97% | 0.9602 |
| Random Forest | 98.15% | 0.9723 |
| Small NN (8, 4) | 98.22% | 0.9825 |

---

## Phase 2: Quantum Pipeline Design
The QML pipeline is built using the **Variational Quantum Classifier (VQC)** framework in Qiskit.

### 1. Quantum Feature Map
- **Type:** `ZZFeatureMap`
- **Qubits:** 4
- **Repetitions (Reps):** 2
- **Purpose:** To capture non-linear, non-convex dependencies between features through entanglement ($R_{ZZ}$ gates), which are difficult for standard linear classical models.

### 2. Variational Ansatz
- **Type:** `RealAmplitudes`
- **Entanglement:** Linear
- **Purpose:** A hardware-efficient circuit that uses parameterized $Y$-rotations to navigate the Hilbert space and find the optimal decision boundary for fraud detection.

### 3. Training & Optimization
- **Optimizer:** `COBYLA` (Constrained Optimization By Linear Approximation)
- **Loss Function:** Cross-Entropy
- **Sampler:** Qiskit Primitives (Sampler) for measuring state probabilities.

---
