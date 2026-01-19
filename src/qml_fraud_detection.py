import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

import os

def load_and_preprocess_data(filename='dataset.csv'):
    # Robust path handling
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    filepath = os.path.join(data_dir, filename)
    
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    
    # 1. Handle Missing Values
    print(f"Original shape: {df.shape}")
    df.dropna(inplace=True)
    print(f"Shape after dropping NaNs: {df.shape}")
    
    # Separate Feature and Target
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    # 2. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Dimensionality Reduction (PCA -> 4 features)
    print("Applying PCA (8 -> 4 features)...")
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    return X_pca, y

def train_classical_baselines(X_train, X_test, y_train, y_test):
    results = {}
    
    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    results['Random Forest'] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }
    
    # 3. MLP (Neural Network)
    print("Training MLP (Classical NN)...")
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]
    results['MLP'] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }
    
    return results

if __name__ == "__main__":
    # Load and Process
    X_pca, y = load_and_preprocess_data()
    
    # Split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train Baselines
    baseline_results = train_classical_baselines(X_train, X_test, y_train, y_test)
    
    # Display Results
    print("\n--- Classical Benchmark Results ---")
    results_df = pd.DataFrame(baseline_results).T
    print(results_df)

    # Save processed data for QML step
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    np.save(os.path.join(data_dir, 'X_train_fraud_detection.npy'), X_train)
    np.save(os.path.join(data_dir, 'X_test_fraud_detection.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_train_fraud_detection.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_test_fraud_detection.npy'), y_test)
    print(f"\nProcessed data saved to {data_dir}")
