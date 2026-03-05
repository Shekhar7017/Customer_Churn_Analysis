# Customer Churn Prediction (Machine Learning Project)

## 📌 Project Overview

Customer churn prediction is a machine learning project that aims to predict whether a telecom customer will leave the service or continue using it.
This is a **Supervised Learning – Classification problem**.

By predicting churn, companies can identify customers who are likely to leave and take preventive actions to retain them.

---

# 📊 Dataset Description

The dataset contains information about telecom customers such as:

* Gender
* SeniorCitizen
* Partner
* Dependents
* Tenure
* PhoneService
* MultipleLines
* InternetService
* OnlineSecurity
* OnlineBackup
* DeviceProtection
* TechSupport
* StreamingTV
* StreamingMovies
* Contract
* PaperlessBilling
* PaymentMethod
* MonthlyCharges
* TotalCharges

### 🎯 Target Variable

**Churn**

* `Yes` → Customer left the company
* `No` → Customer stayed with the company

---

# ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)

---

# 🧠 Machine Learning Workflow

## 1️⃣ Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

# 2️⃣ Load Dataset

```python
df = pd.read_csv("customer_churn.csv")
df.head()
```

---

# 3️⃣ Data Exploration (EDA)

```python
df.info()
df.describe()
df.isnull().sum()
```

This step helped understand:

* dataset structure
* feature types
* missing values

---

# 4️⃣ Fixing Data Types

The **TotalCharges** column was stored as string and converted into numeric.

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
```

Handling missing values:

```python
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
```

---

# 5️⃣ Encoding Categorical Variables

Machine learning models require **numeric input**, so categorical features were encoded.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
```

After encoding, all columns became numeric.

---

# 6️⃣ Train Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Dataset split:

* **80% Training Data**
* **20% Testing Data**

---

# 7️⃣ Detecting Imbalanced Dataset

```python
df['Churn'].value_counts()
```

Example result:

```
No  → 5174
Yes → 1869
```

The dataset is **imbalanced** because one class appears much more than the other.

---

# 8️⃣ Handling Imbalanced Data (SMOTE)

To solve the imbalance problem, **SMOTE (Synthetic Minority Oversampling Technique)** was applied.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

SMOTE creates synthetic samples for the minority class to balance the dataset.

---

# 9️⃣ Model Training (Logistic Regression)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_balanced, y_train_balanced)
```

---

# 🔮 Model Prediction

```python
y_pred = model.predict(X_test)
```

---

# 📈 Model Evaluation

```python
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
```

---

# 📊 Project Pipeline

```
Load Dataset
↓
Data Exploration
↓
Data Cleaning
↓
Encoding
↓
Train-Test Split
↓
SMOTE (Handle Imbalance)
↓
Model Training
↓
Prediction
↓
Evaluation
```

---

# 🚀 Future Improvements

* Try multiple ML models (Random Forest, XGBoost)
* Hyperparameter tuning
* Feature importance analysis
* ROC Curve and AUC evaluation

---

# 👨‍💻 Author

Machine Learning Learning Journey Project
Customer Churn Prediction
