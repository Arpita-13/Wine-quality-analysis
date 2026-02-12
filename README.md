# Wine-quality-analysis
# Machine Learning Classification Models - Streamlit Web Application

## a. Problem Statement
Build and deploy an interactive web application that demonstrates **6 different classification algorithms** on a single dataset. The application should allow users to:
- Upload test data for predictions
- Compare model performance across multiple metrics
- Visualize results through confusion matrices
- Make real-time predictions using trained models

## b. Dataset Description
**Dataset:** Wine Quality (Red Wine) from UCI Machine Learning Repository

| Attribute | Description |
|-----------|------------|
| **Source** | UCI Machine Learning Repository |
| **Instances** | 1,599 wine samples |
| **Features** | 12 physicochemical properties |
| **Target** | Quality score (0-10) → Binary: Good (≥7) vs Not Good (<7) |
| **Class Distribution** | Good: 13.6%, Not Good: 86.4% |

**Features List:**
1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Quality (target)

## c. Models Used and Evaluation Metrics

### Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|---------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.88 | 0.92 | 0.79 | 0.73 | 0.76 | 0.62 |
| Decision Tree | 0.85 | 0.84 | 0.74 | 0.68 | 0.71 | 0.55 |
| KNN | 0.86 | 0.89 | 0.77 | 0.70 | 0.73 | 0.58 |
| Naive Bayes | 0.78 | 0.82 | 0.58 | 0.62 | 0.60 | 0.42 |
| Random Forest | **0.89** | **0.94** | **0.82** | **0.75** | **0.78** | **0.65** |
| XGBoost | 0.88 | 0.93 | 0.81 | 0.72 | 0.76 | 0.63 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|--------------------------------------|
| **Logistic Regression** | Strong baseline performance (88% accuracy). Works well due to reasonable linear separability in scaled features. AUC of 0.92 indicates good ranking capability. |
| **Decision Tree** | Moderate performance with signs of overfitting. Despite max_depth=10, the model struggles to generalize compared to ensemble methods. |
| **KNN** | Good performance (86% accuracy) when features are properly scaled. Captures local patterns effectively but computationally expensive for large datasets. |
| **Naive Bayes** | Lowest performance among all models (78% accuracy). The feature independence assumption doesn't hold for wine quality dataset where features are correlated. |
| **Random Forest** | **Best overall performer** with highest scores across all metrics. Ensemble approach reduces overfitting and captures complex feature interactions effectively. |
| **XGBoost** | Excellent performance comparable to Random Forest. Slightly better precision (81%) but lower recall (72%). Great for imbalanced datasets. |

