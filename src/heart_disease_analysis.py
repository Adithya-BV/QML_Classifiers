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
from sklearn.metrics import accuracy_score, roc_auc_score

import os

def process_heart_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    filepath = os.path.join(data_dir, 'heart_disease_dataset.csv')
    
    print("Loading Heart Disease dataset...")
    df = pd.read_csv(filepath)
    
    # 1. Clean
    df.dropna(inplace=True)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 2. Scale
    print("Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. PCA (13 -> 4 features)
    print("Applying PCA to 4 features...")
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)
    
    # Save for QML
    np.save(os.path.join(data_dir, 'X_train_heart.npy'), X_train)
    np.save(os.path.join(data_dir, 'X_test_heart.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_train_heart.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_test_heart.npy'), y_test)
    
    return X_train, X_test, y_train, y_test

def benchmark_classical(X_train, X_test, y_train, y_test):
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    results['Logistic Regression'] = {
        'Accuracy': accuracy_score(y_test, lr.predict(X_test)),
        'AUC': roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    }
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    results['Random Forest'] = {
        'Accuracy': accuracy_score(y_test, rf.predict(X_test)),
        'AUC': roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    }
    
    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    results['MLP'] = {
        'Accuracy': accuracy_score(y_test, mlp.predict(X_test)),
        'AUC': roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1])
    }
    
    return results

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = process_heart_data()
    results = benchmark_classical(X_train, X_test, y_train, y_test)
    print("\n--- Classical Benchmarks (Heart Disease) ---")
    print(pd.DataFrame(results).T)
