# 📣 Logistic Regression — Social Network Ads (Purchase Prediction)

**Repository:** *Logistic Regression Social Network Ads*
**Notebook:** `Logistic Regression Social network.ipynb`

---

## 🔎 TL;DR

Built an interpretable **Logistic Regression** model to predict whether a user will purchase a product from social network ads using demographic and behavioral features. The notebook demonstrates a **complete ML workflow** — data ingestion, preprocessing, scaling, modeling, evaluation, and visual diagnostics.

---

## 📊 Key Results

* **Dataset:** 400 rows, 5 columns (`User ID`, `Gender`, `Age`, `EstimatedSalary`, `Purchased`)

* **Train/Test split:** 70% / 30% → **Train = 280**, **Test = 120**

* **Model:** `sklearn.linear_model.LogisticRegression()` (default)

* **Test set performance:**

  * **Accuracy:** **0.86**
  * **Classification report (test set):**

    ```
                  precision    recall  f1-score   support

           0       0.83      0.97      0.89        73
           1       0.94      0.68      0.79        47

    accuracy                           0.86       120
    macro avg       0.88      0.83      0.84       120
    weighted avg    0.87      0.86      0.85       120
    ```

* **Confusion Matrix:**
  ![Confusion Matrix](images/confusion%20metrics.png)

* **ROC Curve (AUC \~ 0.91):**
  ![ROC Curve](images/Roc%20Curve.png)

> 📌 **Interpretation:**
> Model is very accurate at identifying non-purchasers (recall = 0.97), but misses \~32% of actual purchasers (recall = 0.68). Depending on business goals, the decision threshold can be adjusted to improve recall.

---

## 📂 Dataset & Preprocessing

* **Data file:** `Social_Network_Ads.csv`
* **Features used (`X`):** `Gender`, `Age`, `EstimatedSalary`
* **Target (`y`):** `Purchased`
* **Steps performed:**

  * Dropped `User ID`
  * Encoded `Gender` (Male = 1, Female = 0)
  * Standardized features using **StandardScaler**
  * Train-test split (70% train, 30% test)

```python
# Encoding gender
df['Gender'] = df['Gender'].apply(lambda x: 1 if str(x).strip().lower() == "male" else 0)

# Feature scaling
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
```

---

## 🧑‍💻 Model Training & Evaluation

* **Logistic Regression model:**

  ```python
  lr = LogisticRegression()
  lr.fit(X_train, y_train)
  y_pred = lr.predict(X_test)
  ```
* **Evaluation metrics generated:**

  * Accuracy score (0.86)
  * Classification report (precision, recall, F1-score)
  * Confusion matrix (visualized above)
  * ROC Curve & AUC

---

## 📈 Insights & Business Relevance

* ✅ **High precision for purchasers (0.94):** When the model predicts a purchase, it’s usually correct. Useful for **targeted campaigns** where false positives are costly.
* ⚠️ **Lower recall for purchasers (0.68):** The model misses \~32% of buyers. If the goal is **maximizing sales capture**, recall should be improved (via class weighting, resampling, or threshold tuning).
* ⚡ **Lightweight & interpretable:** With only three predictors, this model is fast, explainable, and easy to deploy. Ideal for **marketing proof-of-concept**.

---

## 🧾 Reproducibility — How to Run

1. Clone the repo and ensure the dataset `Social_Network_Ads.csv` is present.
2. Install dependencies:

   ```bash
   pip install pandas seaborn matplotlib scikit-learn jupyter
   ```
3. Launch Jupyter and run the notebook:

   ```bash
   jupyter notebook "Logistic Regression Social network.ipynb"
   ```
4. To export plots for README:

   ```python
   plt.savefig("images/confusion metrics.png", bbox_inches="tight")
   plt.savefig("images/Roc Curve.png", bbox_inches="tight")
   ```

---

## 📁 Project Structure

```
├── Logistic Regression Social network.ipynb   # Main notebook
├── Social_Network_Ads.csv                     # Dataset (400 rows)
├── images/                                    # Visualization assets
│   ├── confusion metrics.png
│   └── Roc Curve.png
└── README.md                                  # Documentation
```

---

## 💡 Elevator Pitch

Developed an **86% accurate Logistic Regression model** to predict purchase behavior from social network ads using demographic features (Age, Gender, Estimated Salary). Produced **business-driven insights**: strong precision for targeting campaigns, but opportunities to boost recall for wider sales reach. Delivered a reproducible ML pipeline with visual diagnostics (confusion matrix & ROC curve) for easy deployment.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Jupyter.
