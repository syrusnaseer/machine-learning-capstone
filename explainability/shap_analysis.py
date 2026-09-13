import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("data/insurance.csv")

# Create target
threshold = df["charges"].median()
df["High_Cost"] = (df["charges"] >= threshold).astype(int)

# Features and target
X = df[
    ["age", "sex", "bmi", "children", "smoker", "region"]
]

y = df["High_Cost"]

# Same split used for model comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load final pipeline
pipeline = joblib.load("models/final_model.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
xgb_model = pipeline.named_steps["model"]

# Transform test data
X_test_transformed = preprocessor.transform(X_test)

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

# Feature names
feature_names = preprocessor.get_feature_names_out()

# SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X_test_transformed)

# Global SHAP summary plot
plt.figure()

shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "explainability/shap_global_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Global SHAP analysis completed.")
print("Saved: explainability/shap_global_summary.png")