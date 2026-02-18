# 🏠 Boston Housing Price Prediction

An end-to-end machine learning pipeline for predicting median housing prices in Boston neighborhoods. This project demonstrates the complete data science workflow — from exploratory analysis to model optimization — comparing **10 regression models** and achieving **R² = 0.891** with a tuned LightGBM.

---

## 🎯 Key Results

| Rank | Model | Test R² | CV R² (±std) | RMSE ($1000s) | MAE ($1000s) |
|------|-------|---------|--------------|---------------|--------------|
| 1 | **LightGBM (tuned)** | **0.8910** | 0.8684 ± 0.024 | **2.51** | **1.63** |
| 2 | Gradient Boosting (tuned) | 0.8873 | 0.8720 ± 0.022 | 2.83 | 1.87 |
| 3 | Random Forest (tuned) | 0.8684 | 0.8510 ± 0.045 | 3.18 | 2.01 |

> The best model's predictions are off by approximately **$2,500** on average — a strong result for a dataset with only 506 samples and 13 features.

---

## 📊 Project Workflow

```
Data Loading → EDA → Feature Engineering → Preprocessing → Modeling → Tuning → Evaluation
```

### Exploratory Data Analysis
- Target variable distribution analysis (skewness, outliers)
- Correlation heatmap & multicollinearity detection
- Variance Inflation Factor (VIF) analysis
- Feature-target scatter plots & outlier quantification (IQR method)

### Feature Engineering
- **Log transformation** of target variable to reduce right skewness
- **Interaction features**: LSTAT×RM, NOX×DIS, TAX/RAD ratio
- **Binned categories** for RM and LSTAT with one-hot encoding

### Preprocessing
- **RobustScaler** — chosen for outlier resistance (uses median/IQR instead of mean/std)
- Multicollinearity removal based on VIF analysis
- 80/20 train-test split with shuffle

### Models Compared
| Type | Models |
|------|--------|
| Linear | Linear Regression, Ridge, Lasso, ElasticNet |
| Tree-based | Random Forest, Gradient Boosting, XGBoost, LightGBM |
| Other | SVR, KNN |

### Hyperparameter Tuning
- **GridSearchCV** with 5-fold cross-validation on top 3 models
- Gradient Boosting: 243 parameter combinations
- LightGBM: 1,296 parameter combinations
- Random Forest: 216 parameter combinations

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost, LightGBM |
| Statistics | Statsmodels (VIF analysis) |

---

## 📁 Project Structure

```
boston-housing-price-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   └── boston_housing_analysis.ipynb
│
└── images/
    ├── correlation_heatmap.png
    ├── target_distribution.png
    ├── model_comparison.png
    ├── feature_importance.png
    └── residual_analysis.png
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/tonytyler99/boston-housing-price-prediction.git
cd boston-housing-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook notebooks/boston_housing_analysis.ipynb
```
Or open directly in [Google Colab](https://colab.research.google.com/).

---

## 📌 Key Takeaways

- **Feature engineering matters more than model selection** — log transformation and interaction terms provided ~2-3% R² improvement across all models.
- **Ensemble methods dominate** — tree-based models consistently outperformed linear models, indicating non-linear relationships in the data.
- **RobustScaler was the right choice** — the dataset contains significant outliers (CRIM, ZN, B), and RobustScaler's median/IQR approach handled them well.
- **Ethical consideration** — the 'B' feature encodes racial demographic data, raising fairness concerns for real-world deployment. This feature should be carefully audited or excluded in production.

---

## 📈 Future Improvements

- [ ] Apply SHAP values for model interpretability
- [ ] Experiment with model stacking/blending
- [ ] Build a Streamlit web app for interactive predictions


---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Berkin Kabaağaç**
- GitHub: [@tonytyler99](https://github.com/tonytyler99)
