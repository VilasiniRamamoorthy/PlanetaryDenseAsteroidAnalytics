import os
import json
import streamlit as st
import pandas as pd
import joblib
import sklearn

# --------------------------------
# 1. Page configuration
# --------------------------------

st.set_page_config(
    page_title="Planetary Defense Analytics",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 Planetary Defense Analytics")
st.write(
    "Predict asteroid hazard status and estimate asteroid diameter."
)

# --------------------------------
# 2. Display sklearn version
# --------------------------------

st.sidebar.write(
    "Scikit-learn version:",
    sklearn.__version__
)

# --------------------------------
# 3. Load saved models
# --------------------------------

@st.cache_resource
def load_models():

    pha_model = joblib.load(
        "asteroid_pha_model.pkl"
    )

    diameter_model = joblib.load(
        "best_asteroid_regression_model.pkl"
    )

    return pha_model, diameter_model


try:
    pha_model, diameter_model = load_models()

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# --------------------------------
# 4. Load feature schema
# --------------------------------

if os.path.exists("feature_schema.json"):

    with open("feature_schema.json", "r") as file:
        feature_schema = json.load(file)

else:
    st.warning(
        "Feature schema not found. "
        "Using numeric input fields by default."
    )

    feature_schema = {}


# --------------------------------
# 5. Get model input columns
# --------------------------------

pha_features = list(pha_model.feature_names_in_)

diameter_features = list(
    diameter_model.feature_names_in_
)

all_features = list(
    dict.fromkeys(pha_features + diameter_features)
)


# --------------------------------
# 6. Create user input fields
# --------------------------------

st.subheader("Enter Asteroid Parameters")

user_inputs = {}

with st.form("asteroid_form"):

    for feature in all_features:

        feature_type = feature_schema.get(
            feature, "numeric"
        )

        if feature_type == "numeric":

            user_inputs[feature] = st.number_input(
                f"Enter {feature}",
                value=0.0,
                format="%.6f"
            )

        else:

            user_inputs[feature] = st.text_input(
                f"Enter {feature}",
                value=""
            )

    submit = st.form_submit_button(
        "Predict Asteroid"
    )


# --------------------------------
# 7. Prediction
# --------------------------------

if submit:

    input_df = pd.DataFrame([user_inputs])

    try:

        # Classification input
        pha_input = input_df[pha_features]

        # Regression input
        diameter_input = input_df[diameter_features]

        # PHA prediction
        pha_prediction = pha_model.predict(
            pha_input
        )[0]

        # Diameter prediction
        diameter_prediction = diameter_model.predict(
            diameter_input
        )[0]

        st.subheader("Prediction Results")

        if pha_prediction == 1:
            st.error(
                "⚠️ Potentially Hazardous Asteroid"
            )
        else:
            st.success(
                "✅ Non-Potentially Hazardous Asteroid"
            )

        st.metric(
            "Estimated Asteroid Diameter",
            f"{diameter_prediction:.4f}"
        )

        # Optional probability
        if hasattr(pha_model, "predict_proba"):

            probabilities = pha_model.predict_proba(
                pha_input
            )[0]

            st.write(
                "PHA prediction probabilities:"
            )

            st.write({
                str(label): f"{probability:.2%}"
                for label, probability in zip(
                    pha_model.classes_,
                    probabilities
                )
            })

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )
