import streamlit as st
import pandas as pd
import joblib
import shap

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Insurance Cost Classifier",
    page_icon="🏥",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load("models/final_model.pkl")

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏥 Medical Insurance Cost Classifier")

st.write(
    "Enter the patient's information below to predict "
    "whether their medical insurance charge is High Cost or Low Cost."
)

st.divider()

# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.header("Patient Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# --------------------------------------------------
# Create Input DataFrame
# --------------------------------------------------

input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict", use_container_width=True):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0, 1]

    st.header("Prediction")

    if prediction == 1:
        st.error("⚠️ HIGH COST")
    else:
        st.success("✅ LOW COST")

    st.metric(
        "High Cost Probability",
        f"{probability:.2%}"
    )

    st.divider()

    # --------------------------------------------------
    # SHAP Explanation
    # --------------------------------------------------

    st.header("🔎 SHAP Explanation")

    transformed_input = preprocessor.transform(input_data)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(xgb_model)

    shap_values = explainer.shap_values(transformed_input)

    shap_values = shap_values[0]

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values
    })

    explanation_df["Impact"] = explanation_df["SHAP Value"].apply(
        lambda x: "Higher Cost" if x > 0 else "Lower Cost"
    )

    explanation_df["Absolute Impact"] = (
        explanation_df["SHAP Value"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "Absolute Impact",
        ascending=False
    )

    st.write(
        "Positive SHAP values push the prediction toward High Cost, "
        "while negative values push it toward Low Cost."
    )

    st.dataframe(
        explanation_df[
            ["Feature", "SHAP Value", "Impact"]
        ],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # Input Summary
    # --------------------------------------------------

    st.divider()

    st.subheader("Input Summary")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Capstone Project — Medical Insurance Cost Classification"
)