

# 🔄 ChurnZero — Customer Churn Prediction Pipeline

**Model:** XGBoost with Target Encoding  
**Primary Metric:** PR-AUC (Precision-Recall AUC)  
**Secondary Metric:** F1-Score on Positive (Churn) Class  
**Business Cost:** FN = ₹40,000 (missed churner) | FP = ₹500 (unnecessary retention spend)

---

## 📋 Grading Criteria

| Component | Weight | Notes |
|---|---|---|
| **Model Performance** | 40% | Primary: PR-AUC on held-out test. Secondary: F1 on positive class. FN cost ₹40,000, FP cost ₹500. |
| **Presentation Quality** | 25% | Clarity for exec audience, depth of insight, business framing. |
| **Methodology & Rigour** | 20% | Feature engineering decisions and no data leakage. |
| **Code Quality** | 15% | Reproducibility, readability, organisation, and comments. |

---

## 📁 Pipeline Overview

The pipeline automates the customer churn prediction process:

1. **Data Loading:** Imports training and test datasets.
2. **Preprocessing:** - Drops leaky or redundant columns (e.g., `customer_id`, `balance_decline_percentage`).
    - Separates features and target.
    - Imputes missing values (Median for numeric, Mode for categorical).
3. **Pipeline Construction:** Combines preprocessing with an XGBoost classifier.
4. **Validation:** 5-Fold Stratified Cross-Validation to ensure robust performance.
5. **Optimization:** Performs a search for the optimal decision threshold (F1-Maximising) to handle class imbalance.
6. **Evaluation:** Provides a full report including cost analysis based on business parameters.
7. **Explainability:** Generates SHAP explainability and feature importance plots.
8. **Prediction:** Retrains on the full training set and exports test predictions.

---

## 🔧 Feature Engineering & Model Pipeline

* **Encoding Strategy:** Uses Target Encoding for categorical features to avoid dimensionality explosion and handle high-cardinality features effectively.
* **XGBoost:** Hyperparameters were selected to balance performance and generalisation (n_estimators=500, learning_rate=0.05, max_depth=6).
* **Imbalance Handling:** `scale_pos_weight` is computed to adjust for the class imbalance.

---

## 📈 Visualisations Generated
* `confusion_matrix.png`
* `pr_curve.png`
* `correlation_heatmap.png`
* `shap_summary_plot.png`
* `shap_importance_ranking.png`
* `xgb_feature_importance.png`