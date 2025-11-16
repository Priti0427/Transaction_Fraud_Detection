# 💳 Transaction Fraud Detection

This project builds a **machine learning pipeline** to detect fraudulent financial transactions.  
Multiple classification algorithms are implemented and compared to identify the best-performing model.

---

## 🧠 Overview

Machine learning helps detect fraudulent activity by identifying patterns in transaction data.  
This project demonstrates a full ML workflow — from data preprocessing to model evaluation — both with and without pipelines.

---

## ⚙️ Steps Involved

1. **Data Collection & Cleaning**  
   - Imported dataset from Google Drive  
   - Handled missing values (none found)  
   - Encoded categorical columns

2. **Exploratory Data Analysis (EDA)**  
   - Checked data balance (highly imbalanced: ~8K frauds vs. 6.3M non-frauds)  
   - Visualized correlations and distributions

3. **Feature Engineering**  
   - Selected top features using `SelectKBest` (ANOVA F-test)  
   - Scaled numerical columns with `StandardScaler`

4. **Model Training**  
   - Compared Logistic Regression, Decision Tree, XGBoost, and CatBoost  
   - Used `train_test_split` (70/30) for evaluation

5. **Pipeline Implementation**  
   - Automated preprocessing + feature selection + model training  
   - Created one pipeline per algorithm for fair comparison

---

## 📊 Evaluation Metrics

Each model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- ROC-AUC  

Example:
```python
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), roc_auc_score(y, y_pred)
```

## 🏁 Results Summary

| Model | Accuracy | Notes |
|--------|-----------|-------|
| Logistic Regression | ~96% | Fast baseline |
| Decision Tree | ~99% | High accuracy |
| XGBoost | ~99% | Best balance |
| CatBoost | ~99% | Excellent on large data |

---

## 🧩 Tech Stack

**Language:** Python  

**Libraries:**  
- pandas  
- scikit-learn  
- xgboost  
- catboost  
- matplotlib  
- seaborn  

---

## 🚀 How to Run

```bash
git clone https://github.com/pritisagar/Transaction_Fraud_Detection.git
cd Transaction_Fraud_Detection
pip install -r requirements.txt
python fraud_detection.py
```

# 🧾 Conclusion

This project shows how ML pipelines simplify model building for large-scale fraud detection,
making workflows efficient, consistent, and reusable.

# 📚 Acknowledgements

Scikit-learn Documentation

XGBoost Docs

CatBoost Docs

# 👩‍💻 Author

Priti Sagar
 • https://medium.com/@pritisagar0427
 • https://www.linkedin.com/in/priti-sagar04/
