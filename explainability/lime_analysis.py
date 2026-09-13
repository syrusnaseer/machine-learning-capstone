import pandas as pd
import matplotlib.pyplot as plt
import joblib
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/insurance.csv")

# Create target
threshold = df["charges"].median()
df["High_Cost"] = (df["charges"] >= threshold).astype(int)

# Features and target
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["High_Cost"]

# Convert categorical features to numeric values for LIME
X_encoded = pd.get_dummies(X, columns=["sex", "smoker", "region"])

# Same split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load model
pipeline = joblib.load("models/final_model.pkl")

# Original feature structure expected by the model
def predict_fn(data):
    data_df = pd.DataFrame(data, columns=X_encoded.columns)

    # Convert encoded data back to original categorical values
    result = pd.DataFrame({
        "age": data_df["age"],
        "bmi": data_df["bmi"],
        "children": data_df["children"],
        "sex": data_df[["sex_female", "sex_male"]].idxmax(axis=1).str.replace("sex_", ""),
        "smoker": data_df[["smoker_no", "smoker_yes"]].idxmax(axis=1).str.replace("smoker_", ""),
        "region": data_df[
            [
                "region_northeast",
                "region_northwest",
                "region_southeast",
                "region_southwest"
            ]
        ].idxmax(axis=1).str.replace("region_", "")
    })

    return pipeline.predict_proba(result)


# Create LIME explainer
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_encoded.columns.tolist(),
    class_names=["Low Cost", "High Cost"],
    mode="classification",
    random_state=42
)

# Generate explanations for 3 patients
for i in range(3):

    explanation = explainer.explain_instance(
        X_test.iloc[i].values,
        predict_fn,
        num_features=8
    )

    figure = explanation.as_pyplot_figure()

    plt.tight_layout()

    filename = f"explainability/lime_patient_{i + 1}.png"

    figure.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"Saved: {filename}")

print("LIME analysis completed.")