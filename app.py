import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# CONFIG
st.set_page_config(
    page_title="House Price Category Predictor",
    layout="wide"
)

MODEL_ACCURACY = 0.53

# LOAD COMPONENTS
@st.cache_resource
def load_components():
    model = joblib.load("best_house_price_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, label_encoder, feature_columns

model, scaler, label_encoder, FEATURE_COLUMNS = load_components()

# FEATURE ENGINEERING + TRACE OUTPUT
def prepare_input(user_input: dict):
    df = pd.DataFrame([user_input])

    # ---------- FEATURE ENGINEERING ----------
    df["house_age"] = 1900 - df["yr_built"] 
    df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["price_per_sqft"] = df["price"] / (df["sqft_living"] + 1)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["sqft_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1)
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)

    # Save derived features for display (TRACEABILITY)
    derived_features = df[
        [
            "house_age",
            "is_renovated",
            "price_per_sqft",
            "total_rooms",
            "sqft_ratio",
            "has_basement",
        ]
    ].copy()

    # Drop replaced columns
    df.drop(columns=["yr_built", "yr_renovated"], inplace=True)

    # Ensure same columns as training
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_COLUMNS]
    scaled = scaler.transform(df)

    return scaled, derived_features

# UI
st.title("House Price Category Prediction")
st.write(
    "This application predicts whether a house falls into "
    "Budget, Affordable, Premium, or Luxury categories."
)

# SIDEBAR INPUTS
st.sidebar.header("House Details")

user_input = {
    "bedrooms": st.sidebar.number_input(
        "Bedrooms: (1 - 5)", min_value=1, max_value=5, value=1, step=1
    ),
    "bathrooms": st.sidebar.number_input(
        "Bathrooms: (1 - 5)", min_value=1, max_value=10, value=1, step=1
    ),
    "sqft_living": st.sidebar.number_input(
        "Living Area (sqft): (100 - 5000)", min_value=100, max_value=5000, value=1000
    ),
    "sqft_lot": st.sidebar.number_input(
        "Lot Size (sqft): (100 - 200000)", min_value=100, max_value=200000, value=5000
    ),
    "floors": st.sidebar.number_input(
        "Floors: (1 - 10)", min_value=1, max_value=10, value=1, step=1
    ),
    "waterfront": st.sidebar.selectbox(
        "Do the house near the waterfront?: (0 for No and 1 for Yes)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
    ),
    "view": st.sidebar.slider("View Rating (0–4)", 0, 4, 0),
    "condition": st.sidebar.slider("Condition (1–5)", 1, 5, 3),
    "sqft_above": st.sidebar.number_input(
        "Above Ground Area (sqft): (100 - 5000)", min_value=100, max_value=5000, value=1500
    ),
    "sqft_basement": st.sidebar.number_input(
        "Basement Area (sqft): (0 - 2000)", min_value=0, max_value=2000, value=0
    ),
    "price": st.sidebar.number_input(
        "House Price ($): (10000 - 2000000)", min_value=10000, max_value=2000000, value=500000
    ),
    "yr_built": st.sidebar.number_input(
        "Year Built: (1900 - 2025)", min_value=1900, max_value=2025, value=1900
    ),
    "yr_renovated": st.sidebar.number_input(
        "Year Renovated: (0 if not - 2025)", min_value=0, max_value=2025, value=0
    ),
}

# PREDICTION
if st.button("Predict Price Category"):
    X_scaled, derived_features = prepare_input(user_input)

    pred_encoded = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]
    prediction = label_encoder.inverse_transform([pred_encoded])[0]

    confidence = np.max(pred_proba) * 100

    # RESULTS
    st.success(
        f"Predicted Category: {prediction}\n\n"
        f"Confidence: {confidence:.2f}%"
    )

    st.info(
        f"Model Accuracy (Test Set): {MODEL_ACCURACY * 100:.2f}%"
    )

    # DERIVED FEATURES (TRACEABILITY)
    st.subheader("Derived Features Used by the Model")

    display_df = derived_features.T
    display_df.columns = ["Value"]

    display_df.loc["is_renovated", "Value"] = (
        "Yes" if display_df.loc["is_renovated", "Value"] == 1 else "No"
    )
    display_df.loc["has_basement", "Value"] = (
        "Yes" if display_df.loc["has_basement", "Value"] == 1 else "No"
    )

    st.table(display_df)

    # PROBABILITY CHART
    fig, ax = plt.subplots()

    probabilities_percent = pred_proba * 100
    bars = ax.bar(label_encoder.classes_, probabilities_percent)

    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Prediction Probability Distribution")

    for bar, label in zip(bars, label_encoder.classes_):
        bar.set_alpha(1.0 if label == prediction else 0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )
    st.pyplot(fig)

# FOOTER
st.markdown(
    """
    ---
    Category Interpretation
    - Budget: Lowest 25% of house prices  
    - Affordable: 25 – 50% range  
    - Premium: 50 – 75% range  
    - Luxury: Top 25% of house prices  
    """
)
