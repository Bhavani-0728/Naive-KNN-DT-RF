import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from pandas.api.types import is_numeric_dtype

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Pass / Fail Prediction")
st.write("Predict student success using key academic & behavioural factors")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("student-mat.csv")

df_raw = load_data()

# ---------------- RAW DATA ----------------
st.subheader("📌 Raw Dataset Preview")
st.dataframe(df_raw.head())

# ---------------- DATA CLEANING ----------------
df = df_raw.copy()

df.drop_duplicates(inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

df["Pass"] = df["G3"].apply(lambda x: "Yes" if x >= 10 else "No")
df.drop("G3", axis=1, inplace=True)

# Remove low-impact features
columns_to_remove = ["Mother_job", "Father_job"]
df.drop(columns=[col for col in columns_to_remove if col in df.columns],
        inplace=True)

# ---------------- ENCODING ----------------
encoders = {}

for col in df.columns:
    if not is_numeric_dtype(df[col]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# ---------------- FEATURES ----------------
X = df.drop("Pass", axis=1)
y = df["Pass"]

# ---------------- FULL MODEL (FOR IMPORTANCE) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_full = RandomForestClassifier(random_state=42)
model_full.fit(X_train, y_train)

# ---------------- FEATURE SELECTION ----------------
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model_full.feature_importances_
}).sort_values(by="Importance", ascending=False)

TOP_FEATURES = importance_df["Feature"].head(10).tolist()

# Ensure studytime is included
if "weekly_studytime" not in TOP_FEATURES:
    TOP_FEATURES[-1] = "weekly_studytime"

# ---------------- CLEANED DATA DISPLAY ----------------
st.subheader("✅ Cleaned Dataset Preview (Selected Features)")
st.dataframe(df[TOP_FEATURES].head())

# ---------------- REDUCED DATA ----------------
X_reduced = X[TOP_FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ---------------- PERFORMANCE ----------------
st.subheader("📊 Model Performance")

st.metric("Accuracy", f"{accuracy:.2f}")

st.write("### Confusion Matrix")

fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax)
st.pyplot(fig)

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📈 Feature Importance")

fig2, ax2 = plt.subplots()
ax2.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
ax2.invert_yaxis()
st.pyplot(fig2)

# ---------------- USER INPUT (SLIDER UI) ----------------
st.subheader("🧮 Enter Student Details")

input_data = {}
cols = st.columns(2)

for i, feature in enumerate(TOP_FEATURES):
    with cols[i % 2]:

        # ---------------- CATEGORICAL ----------------
        if feature in encoders:
            options = encoders[feature].classes_
            selected = st.selectbox(feature, options)
            input_data[feature] = encoders[feature].transform([selected])[0]

        # ---------------- NUMERIC ----------------
        else:

            if feature == "failures":
                input_data[feature] = st.slider(feature, 0, 4, 0)

            elif feature == "absences":
                input_data[feature] = st.slider(feature, 0, 30, 5)

            elif feature in ["G1", "G2"]:
                input_data[feature] = st.slider(feature, 0, 20, 10)

            elif feature == "weekly_studytime":
                study_map = {
                    1: "< 2 hours",
                    2: "2 – 5 hours",
                    3: "5 – 10 hours",
                    4: "> 10 hours"
                }

                study_value = st.slider(
                    "Weekly Study Time",
                    1, 4, 2,
                    format="%d"
                )

                st.caption(f"Selected: {study_map[study_value]}")
                input_data[feature] = study_value

            elif feature in ["health", "goout"]:
                input_data[feature] = st.slider(feature, 1, 5, 3)

            else:
                min_val = int(X_reduced[feature].min())
                max_val = int(X_reduced[feature].max())
                default = int(X_reduced[feature].mean())

                input_data[feature] = st.slider(feature, min_val, max_val, default)

input_df = pd.DataFrame([input_data])

# ---------------- PREDICTION ----------------
if st.button("Predict Outcome"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    confidence = np.max(probabilities)

    if prediction == 1:
        st.success("Prediction: PASS ✅")
    else:
        st.error("Prediction: FAIL ❌")

    st.write(f"Confidence: {confidence:.2f}")
