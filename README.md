# QML-Classifiers

## Phase 1: Data Pre-processing & Classical Benchmarking (Weeks 1-2)

### Data Strategy
- **Constraint:** Limited to 4 features to accommodate a 4-qubit NISQ system.
- **Selection:** Used Random Forest feature importance to select `ratio_to_median_purchase_price`, `online_order`, `distance_from_home`, and `used_pin_number`.
- **Cleaning:** Handled extreme outliers in transaction distances and price ratios to ensure model stability.

### Classical Performance Table
| Model | Accuracy | AUC-ROC |
| :--- | :--- | :--- |
| Logistic Regression | 95.97% | 0.9602 |
| Random Forest | 98.15% | 0.9723 |
| Small NN (8, 4) | 98.22% | 0.9825 |

**Observation:** The high performance of the Random Forest and MLP models indicates a strong signal in the data, but the non-linear boundaries (seen in the pairplot) suggest that a Quantum Variational Circuit may find a unique high-dimensional mapping.
