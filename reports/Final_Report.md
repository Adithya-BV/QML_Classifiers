# QML Project Report

**Student Name:** Bollu Venkata Adithya  
**Student ID:** 24116025  
**Project:** Quantum Machine Learning for Classification

---

## 1. Objective
The primary objective of this project is to design, implement, and benchmark a **Hybrid Quantum-Classical Classifier** for noisy, non-convex classification tasks. The goal is to evaluate if Quantum Machine Learning (QML) models, specifically Variational Quantum Circuits (VQCs), can offer competitive performance or advantages over classical baselines (Logistic Regression, Random Forest) on high-dimensional datasets when subjected to dimensionality reduction.

## 2. Overview
This project explores the application of QML on two distinct datasets:
1.  **Credit Card Fraud Detection**: A highly imbalanced, high-volume dataset.
2.  **Heart Disease Prediction**: A medical diagnostic dataset with complex feature interactions.

We utilized **PennyLane** for quantum circuit simulation and **PyTorch** for constructing the hybrid neural network architecture. The analysis includes data preprocessing, classical benchmarking, hybrid model training, and a final comparative assessment using Area Under the ROC Curve (AUC) and Accuracy metrics. A bonus investigation into **Data Re-uploading Quantum Neural Networks (QNNs)** was also conducted to explore pure quantum architectures.

## 3. Data Analysis and Preprocessing
To adapt the classical datasets for a **4-qubit** quantum simulation, the following pipeline was applied:

### Fraud Detection Dataset
*   **Original Dimensions**: 8 features.
*   **Cleaning**: Removed rows with missing values.
*   **Scaling**: Applied `StandardScaler` for zero mean and unit variance.
*   **Dimensionality Reduction**: Utilized **Principal Component Analysis (PCA)** to compress 8 features into **4 principal components**, retaining ~59% of the variance.
*   **Splitting**: 80/20 Train-Test split, stratified by the target label.

### Heart Disease Dataset
*   **Original Dimensions**: 13 features.
*   **Preprocessing**: similar pipeline (Cleaning, Scaling).
*   **Dimensionality Reduction**: PCA reduced 13 features to **4 components** (retaining ~51% variance). This aggressive reduction posed a challenge for feature retention.

## 4. Quantum Circuit Design
The core of the solution is a Variational Quantum Circuit (VQC) with two main blocks:

1.  **Feature Map (Encoding)**:
    *   **Type**: `qml.AngleEmbedding`
    *   **Function**: Encodes the 4 classical features into the rotation angles of qubits (e.g., $R_x(\theta)$), mapping classical data to a quantum state.
2.  **Ansatz (Variational Layers)**:
    *   **Type**: `qml.StronglyEntanglingLayers`
    *   **Depth**: 2 layers.
    *   **Function**: Applies a series of parameterized rotations and CNOT entangling gates. These parameters are optimized during training to minimize the cost function.

**Visualization**:
```
[AngleEmbedding] --> [StronglyEntanglingLayers] --> [Measurement (PauliZ)]
```

**(Bonus) Data Re-uploading QNN**:
*   Implemented a pure QNN where the encoding and variational layers are interleaved (repeated 4 times) to increase model expressivity without adding qubits.

## 5. Training Pipeline
A **Hybrid Quantum-Classical** training loop was established:
*   **Framework**: PyTorch (`qml.qnn.TorchLayer`).
*   **Model**:
    *   *Quantum Layer*: 4-qubit VQC outputting 4 expectation values.
    *   *Classical Post-processing*: A Linear Layer (`nn.Linear`) mapping 4 quantum outputs to 1 prediction, followed by a Sigmoid activation.
*   **Optimizer**: **Adam** (Learning Rate = 0.01).
*   **Loss Function**: Binary Cross Entropy (`nn.BCELoss`).
*   **Training**: Tuned for convergence (2-10 epochs) depending on dataset size.

## 6. Comparative Analysis

### Dataset 1: Fraud Detection
| Model | Accuracy | AUC |
|-------|----------|-----|
| Logistic Regression (Baseline) | 91.35% | 0.8812 |
| Random Forest (Baseline) | 99.85% | 0.9999 |
| **Hybrid QML (Proposed)** | **97.03%** | **0.9678** |

*Analysis*: The Hybrid QML model significantly outperformed the linear baseline (Logistic Regression) and achieved a high AUC (>0.96), demonstrating that the quantum features captured non-linear patterns effectively.

### Dataset 2: Heart Disease
| Model | Accuracy | AUC |
|-------|----------|-----|
| Logistic Regression | 83.90% | 0.9253 |
| Random Forest | 100.00% | 1.0000 |
| **Hybrid QML** | **67.32%** | **0.7810** |

*Analysis*: The QML model underperformed on this dataset. The primary factor is the aggressive PCA reduction (13 -> 4 features) losing ~49% of the information. While the classical Random Forest could recover (likely due to its ensemble nature and handling of remaining features), the shallow quantum circuit struggled with the information loss.

### Robustness Check
*   **Noisy Simulation**: A `DepolarizingChannel` (p=5%) was applied to the Heart Disease model. The pipeline remained stable, though performance degraded as expected, validating the simulation setup.

## 7. Conclusion
This project successfully demonstrated the end-to-end implementation of a Hybrid QML classifier.
1.  **Success**: On the Fraud Detection dataset, the QML model proved highly effective, rivaling classical deep learning performance.
2.  **Challenge**: The Heart Disease analysis highlighted the critical dependency of NISQ-era QML (limited qubits) on effective dimensionality reduction; losing too much variance during PCA severely hampers quantum learning.
3.  **Future**: The Bonus Data Re-uploading architecture suggests a path forward to improve expressivity without increasing qubit count.

The deliverables (Notebook, Source Code, and this Report) verify the completion of all project requirements.
