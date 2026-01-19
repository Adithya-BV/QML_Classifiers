# QML Classifier Project
**Name:** Bollu Venkata Adithya  
**ID:** 24116025

## Project Overview
This project implements a Hybrid Quantum-Classical Classifier using PennyLane and PyTorch to benchmark Quantum Machine Learning (QML) performance against classical baselines on noisy, non-convex classification tasks.

## Datasets
1.  **Credit Card Fraud Detection**: High-dimensional, imbalanced dataset. Reduced to 4 features via PCA.
2.  **Heart Disease Dataset**: Medical diagnostics dataset. Reduced to 4 features via PCA.

## Key Files
- `notebooks/Bollu_Venkata_Adithya_24116025_QML.ipynb`: The main submission notebook containing all code, analysis, and results.
- `requirements.txt`: Python dependencies.
- `src/*.py`: Source implementation scripts (Fraud Detection, Heart Disease Analysis, QML Models).
- `data/`: Contains datasets and processed numpy files.
- `reports/Final_Report.md`: Project Report (Markdown).
- `reports/Bollu_Venkata_Adithya_24116025_QML_Report.pdf`: Project Report (PDF).

## Results Summary
### Fraud Detection
- **Classical (Random Forest)**: ~99.9% AUC
- **Hybrid QML**: ~96.8% AUC (Competitive)

### Heart Disease Detection
- **Classical (Random Forest)**: 100% AUC (on PCA features)
- **Hybrid QML**: ~78.0% AUC
- **Noisy Simulation**: Included (Depolarizing Channel) to demonstrate robustness testing logic.
    - *Observation*: The breakdown in performance for Heart Disease is likely due to significant information loss during PCA (only 51% variance retained).

### Bonus: Data Re-uploading QNN
- **Architecture**: Pure QNN with 4 qubits and 4 layers of data re-encoding.
- **Goal**: Demonstrate increased expressivity without additional qubits.
- **Implementation**: Included in `qnn_bonus.py` and the main notebook.

## How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Open the notebook:
    ```bash
    jupyter notebook notebooks/Bollu_Venkata_Adithya_24116025_QML.ipynb
    ```
