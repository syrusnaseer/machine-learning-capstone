import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/insurance.csv")

# Create target
threshold = df["charges"].median()
df["High_Cost"] = (df["charges"] >= threshold).astype(int)

# Features and target
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["High_Cost"]

# Same train/test split used for the models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load final model
pipeline = joblib.load("models/final_model.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
xgb_model = pipeline.named_steps["model"]

# Transform test data
X_test_transformed = preprocessor.transform(X_test)

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()

# SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed)

# Generate explanations for 3 individual patients
for i in range(3):
    plt.figure()

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[i],
            base_values=explainer.expected_value,
            data=X_test_transformed[i],
            feature_names=feature_names
        ),
        show=False
    )

    plt.tight_layout()

    filename = f"explainability/shap_local_patient_{i + 1}.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")

print("Local SHAP analysis completed.")