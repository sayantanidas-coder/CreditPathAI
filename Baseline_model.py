import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib
import os


print("[INFO] Loading cleaned dataset...")
df = pd.read_csv("data/loan_data_clean.csv")
print("[INFO] Dataset loaded successfully!")
print(df.head())
print(f"[INFO] Shape: {df.shape}")


num_features = ["LoanAmount", "InterestRate", "DTIRatio", "CreditScore", "MonthsEmployed"]
cat_features = [col for col in df.columns if df[col].dtype == 'object']
print("\n[INFO] Numerical Features:", num_features)
print("[INFO] Categorical Features:", cat_features)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

model = LogisticRegression(max_iter=1000)


clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])


X = df.drop("Default", axis=1)
y = df["Default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n[INFO] Data successfully split into train and test sets")


print("\n[INFO] Training Logistic Regression Baseline Model...")
clf.fit(X_train, y_train)
print("[INFO] Model trained successfully!")


y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


print("\nBASELINE MODEL PERFORMANCE:-")
print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1 Score        : {f1:.4f}")
print(f"AUC-ROC Score   : {auc:.4f}")
print("\nConfusion Matrix:")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")


os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/logistic_baseline.pkl")
print("\n[INFO] Baseline model saved to models/logistic_baseline.pkl")


