# 📊 Customer Churn Prediction (Machine Learning Project)

## 📌 Project Overview

Customer churn prediction is a machine learning project that aims to
predict whether a telecom customer will leave the service or continue
using it.

This is a **Supervised Machine Learning -- Classification problem**.

By predicting churn, companies can identify customers who are likely to
leave and take preventive actions to retain them.

------------------------------------------------------------------------

# 📊 Dataset Description

The dataset contains telecom customer information such as:

-   Gender
-   SeniorCitizen
-   Partner
-   Dependents
-   Tenure
-   PhoneService
-   MultipleLines
-   InternetService
-   OnlineSecurity
-   OnlineBackup
-   DeviceProtection
-   TechSupport
-   StreamingTV
-   StreamingMovies
-   Contract
-   PaperlessBilling
-   PaymentMethod
-   MonthlyCharges
-   TotalCharges

### 🎯 Target Variable

**Churn**

Yes → Customer left the company\
No → Customer stayed with the company

------------------------------------------------------------------------

# ⚙️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Imbalanced-learn (SMOTE)

------------------------------------------------------------------------

# 🧠 Machine Learning Workflow

## 1️⃣ Import Libraries

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

------------------------------------------------------------------------

# 2️⃣ Load Dataset

``` python
df = pd.read_csv("customer_churn.csv")
df.head()
```

------------------------------------------------------------------------

# 3️⃣ Data Exploration (EDA)

``` python
df.info()
df.describe()
df.isnull().sum()
```

This step helped understand:

-   dataset structure
-   feature types
-   missing values

------------------------------------------------------------------------

# 4️⃣ Data Cleaning

The **TotalCharges** column was stored as string and converted into
numeric.

``` python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
```

Handling missing values:

``` python
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
```

------------------------------------------------------------------------

# 5️⃣ Encoding Categorical Variables

Machine learning models require **numeric input**, so categorical
features were encoded.

``` python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
```

------------------------------------------------------------------------

# 6️⃣ Train Test Split

``` python
from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Dataset split:

-   80% Training Data
-   20% Testing Data

------------------------------------------------------------------------

# 7️⃣ Handling Imbalanced Data (SMOTE)

Customer churn datasets are usually **imbalanced**, meaning one class
has more samples than the other.

To fix this, we used **SMOTE (Synthetic Minority Oversampling
Technique)**.

``` python
from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

------------------------------------------------------------------------

# 8️⃣ Model Training (Logistic Regression)

``` python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    solver='lbfgs',
    max_iter=5000,
    class_weight='balanced'
)

model.fit(X_train, y_train)
```

------------------------------------------------------------------------

# 9️⃣ Model Testing

``` python
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
```

------------------------------------------------------------------------

# 📈 Model Evaluation

``` python
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))
```

### 📊 Logistic Regression Results

Accuracy: **75%**\
ROC-AUC Score: **0.86**

------------------------------------------------------------------------

# 🌳 Decision Tree Model

``` python
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:,1]
```

### 📊 Decision Tree Results

Accuracy: **73%**\
ROC-AUC Score: **0.76**

------------------------------------------------------------------------

# 🌲 Random Forest Model

``` python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:,1]
```

### 📊 Random Forest Results

Accuracy: **80%**\
ROC-AUC Score: **0.83**

------------------------------------------------------------------------

# 📊 Model Comparison

  Model                 Accuracy   ROC-AUC
  --------------------- ---------- ---------
  Logistic Regression   75%        0.86
  Decision Tree         73%        0.76
  Random Forest         **80%**    0.83

**Random Forest achieved the highest accuracy, while Logistic Regression
achieved the best ROC-AUC score.**

------------------------------------------------------------------------

# 📊 Project Pipeline

Load Dataset\
↓\
Data Exploration\
↓\
Data Cleaning\
↓\
Encoding\
↓\
Train-Test Split\
↓\
SMOTE (Handle Imbalance)\
↓\
Model Training\
↓\
Model Evaluation\
↓\
Model Comparison

------------------------------------------------------------------------

# 🚀 Future Improvements

-   Hyperparameter tuning
-   Feature importance analysis
-   ROC Curve visualization
-   Model deployment using Flask / Streamlit

------------------------------------------------------------------------

# 👨‍💻 Author

Machine Learning Learning Journey Project\
Customer Churn Prediction
