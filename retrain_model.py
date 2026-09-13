import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("data/insurance.csv")


# Create target
threshold = df["charges"].median()
df["High_Cost"] = (df["charges"] >= threshold).astype(int)


# Features and target
X = df.drop(columns=["charges", "High_Cost"])
y = df["High_Cost"]


# Same train/test split as the notebook
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Feature groups
categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]


# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])


# XGBoost model
xgboost_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    ))
])


# Train
xgboost_model.fit(X_train, y_train)


# Test
predictions = xgboost_model.predict(X_test)
probabilities = xgboost_model.predict_proba(X_test)[:, 1]


# Save compatible model
joblib.dump(
    xgboost_model,
    "models/final_model.pkl"
)


print("Model trained successfully.")
print("Threshold:", threshold)
print("Test predictions:", len(predictions))
print("Model saved to: models/final_model.pkl")