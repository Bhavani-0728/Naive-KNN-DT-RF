import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.datasets import load_iris, load_diabetes

# --------------------------------------------------
# Page Config (static, must be first)
# --------------------------------------------------
st.set_page_config(page_title="ML App", layout="wide")

# --------------------------------------------------
# Dark UI
# --------------------------------------------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Configuration")

model_name = st.sidebar.selectbox(
    "Select Model",
    (
        "Naive Bayes (Classification)",
        "KNN - Classification",
        "KNN - Regression",
        "Decision Tree - Classification",
        "Decision Tree - Regression"
    )
)

use_grid = st.sidebar.checkbox("Use GridSearchCV")

st.sidebar.markdown("---")
st.sidebar.markdown("📂 Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --------------------------------------------------
# Dynamic Title
# --------------------------------------------------
model_titles = {
    "Naive Bayes (Classification)": "Naive Bayes Classifier",
    "KNN - Classification": "K-Nearest Neighbors (Classification)",
    "KNN - Regression": "K-Nearest Neighbors (Regression)",
    "Decision Tree - Classification": "Decision Tree Classifier",
    "Decision Tree - Regression": "Decision Tree Regressor"
}

st.markdown(
    f"<title>{model_titles[model_name]}</title>",
    unsafe_allow_html=True
)

st.title(f"🧠 {model_titles[model_name]}")
task_label = "Classification" if "Classification" in model_name else "Regression"
st.caption(f"📌 Task Type: {task_label}")

# --------------------------------------------------
# Logger
# --------------------------------------------------
def log(msg):
    st.write(f"🔹 {msg}")

# --------------------------------------------------
# Dataset Handling
# --------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset loaded")

    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    task = "regression" if y.dtype != "int" else "classification"
    log("Using uploaded dataset")

else:
    if "Regression" in model_name:
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        task = "regression"
    else:
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        task = "classification"

    log("Using default sklearn dataset")

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

log("Train-test split completed")

# --------------------------------------------------
# Model Selection
# --------------------------------------------------
if model_name == "Naive Bayes (Classification)":
    model = GaussianNB()

elif model_name == "KNN - Classification":
    k = st.sidebar.slider("K (Neighbors)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)

elif model_name == "KNN - Regression":
    k = st.sidebar.slider("K (Neighbors)", 1, 15, 5)
    model = KNeighborsRegressor(n_neighbors=k)

elif model_name == "Decision Tree - Classification":
    depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)

elif model_name == "Decision Tree - Regression":
    depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)

# --------------------------------------------------
# GridSearchCV Params
# --------------------------------------------------
param_grids = {
    "Naive Bayes (Classification)": {
        "var_smoothing": [1e-9, 1e-8, 1e-7]
    },
    "KNN - Classification": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"]
    },
    "KNN - Regression": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"]
    },
    "Decision Tree - Classification": {
        "max_depth": [3, 5, 10, None],
        "criterion": ["gini", "entropy"]
    },
    "Decision Tree - Regression": {
        "max_depth": [3, 5, 10, None],
        "criterion": ["squared_error", "absolute_error"]
    }
}

# --------------------------------------------------
# Train Button
# --------------------------------------------------
if st.button("🚀 Train Model"):

    if use_grid:
        log("GridSearchCV started")

        grid = GridSearchCV(
            model,
            param_grids[model_name],
            cv=5,
            scoring="accuracy" if task == "classification" else "neg_mean_squared_error",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        st.success("GridSearchCV completed")
        st.json(grid.best_params_)
        log(f"Best Params: {grid.best_params_}")

    else:
        model.fit(X_train, y_train)
        log("Model trained without GridSearchCV")

    y_pred = model.predict(X_test)

    st.markdown("## 📊 Evaluation")

    # ---------------- CLASSIFICATION ----------------
    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Accuracy: {acc:.4f}")

        st.markdown("### 📈 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown("### 🧾 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    # ---------------- REGRESSION ----------------
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.success(f"MSE: {mse:.4f}")
        st.success(f"RMSE: {rmse:.4f}")
        st.success(f"R² Score: {r2:.4f}")

    # ---------------- DT VISUALIZATION ----------------
    if "Decision Tree" in model_name:
        st.markdown("## 🌲 Decision Tree Visualization")

        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=X.columns,
            filled=True,
            ax=ax
        )
        st.pyplot(fig)

    # ---------------- SAVE MODEL ----------------
    st.markdown("## 💾 Save Trained Model")
    if st.button("Save Trained Model"):
        with open("trained_model.pkl", "wb") as f:
            pickle.dump(model, f)
        st.success("Model saved as trained_model.pkl")