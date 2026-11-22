
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Asteroid Prediction System",
    page_icon="☄️",
    layout="wide"
)

st.title("☄️ Asteroid Prediction System")

st.write(
    "Enter asteroid physical and orbital parameters "
    "to predict its hazardous status and estimated diameter."
)

# --------------------------------
# 2. Load Saved Pipelines
# --------------------------------

MODEL_DIR = Path(__file__).parent / "models"

@st.cache_resource
def load_models():

    classification_model = joblib.load(
        MODEL_DIR / "case1_pha_pipeline.pkl"
    )

    regression_model = joblib.load(
        MODEL_DIR / "case2_diameter_pipeline.pkl"
    )

    return classification_model, regression_model


try:
    classification_model, regression_model = load_models()

except Exception as e:
    st.error(f"Unable to load saved pipelines: {e}")
    st.stop()


# --------------------------------
# 3. Get Required Features
# --------------------------------

def get_features(model):

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Pipeline does not contain feature names. "
            "Save the pipeline after fitting it with a pandas DataFrame."
        )

    return list(model.feature_names_in_)


try:
    classification_features = get_features(
        classification_model
    )

    regression_features = get_features(
        regression_model
    )

except ValueError as e:
    st.error(str(e))
    st.stop()


# Combine features needed by both models
all_features = list(dict.fromkeys(
    classification_features + regression_features
))


# --------------------------------
# 4. Input Form
# --------------------------------

st.subheader("Enter Asteroid Parameters")

st.info(
    "Enter all required parameters. Use the same units "
    "and definitions as those used during model training."
)

with st.form("asteroid_input_form"):

    input_values = {}

    # Display inputs in three columns
    columns = st.columns(3)

    for index, feature in enumerate(all_features):

        with columns[index % 3]:

            input_values[feature] = st.number_input(
                label=feature,
                value=0.0,
                step=0.01,
                format="%.6f",
                key=f"input_{feature}"
            )

    submitted = st.form_submit_button(
        "Predict Asteroid Status and Diameter",
        use_container_width=True
    )


# --------------------------------
# 5. Validate Inputs
# --------------------------------

def validate_inputs(values):

    errors = []

    for feature, value in values.items():

        if not np.isfinite(value):
            errors.append(
                f"{feature} must contain a valid finite number."
            )

    # Physical parameter validation
    if "diameter" in values and values["diameter"] < 0:
        errors.append("Diameter cannot be negative.")

    if "albedo" in values:
        if not 0 <= values["albedo"] <= 1:
            errors.append("Albedo must be between 0 and 1.")

    if "e" in values:
        if not 0 <= values["e"] < 1:
            errors.append(
                "Orbital eccentricity must be between 0 and 1."
            )

    if "a" in values and values["a"] <= 0:
        errors.append("Semi-major axis must be greater than zero.")

    return errors


# --------------------------------
# 6. Make Predictions
# --------------------------------

if submitted:

    errors = validate_inputs(input_values)

    if errors:

        st.error("Please correct the following inputs:")

        for error in errors:
            st.write(f"- {error}")

    else:

        # Create input DataFrame
        input_df = pd.DataFrame(
            [input_values]
        )

        try:

            # Prepare separate inputs for each pipeline
            classification_input = input_df[
                classification_features
            ]

            regression_input = input_df[
                regression_features
            ]

            # Case 1: PHA Classification
            pha_prediction = classification_model.predict(
                classification_input
            )[0]

            # Case 2: Diameter Regression
            diameter_prediction = regression_model.predict(
                regression_input
            )[0]

            st.divider()

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            # Classification result
            with col1:

                st.markdown("### Case 1: PHA Classification")

                if str(pha_prediction).strip().lower() in [
                    "1", "true", "yes"
                ]:

                    st.error(
                        "Potentially Hazardous Asteroid (PHA)"
                    )

                elif str(pha_prediction).strip().lower() in [
                    "0", "false", "no"
                ]:

                    st.success(
                        "Non-Potentially Hazardous Asteroid"
                    )

                else:
                    st.info(
                        f"Predicted PHA class: {pha_prediction}"
                    )

                # Display classification probability
                if hasattr(
                    classification_model,
                    "predict_proba"
                ):

                    probabilities = (
                        classification_model.predict_proba(
                            classification_input
                        )[0]
                    )

                    classes = classification_model.classes_

                    for cls, probability in zip(
                        classes, probabilities
                    ):

                        if str(cls) == "1":
                            st.metric(
                                "PHA Probability",
                                f"{probability * 100:.2f}%"
                            )

            # Regression result
            with col2:

                st.markdown("### Case 2: Diameter Prediction")

                if (
                    np.isfinite(diameter_prediction)
                    and diameter_prediction >= 0
                ):

                    st.metric(
                        "Predicted Asteroid Diameter",
                        f"{diameter_prediction:.4f}"
                    )

                    st.caption(
                        "Diameter unit: same unit used during training."
                    )

                else:
                    st.error(
                        "The model returned an invalid diameter."
                    )

            # Display entered values
            with st.expander("View Input Parameters"):

                st.dataframe(
                    input_df,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Prediction failed. Please check the model "
                f"features and input values. Details: {e}"
            )
