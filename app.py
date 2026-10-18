"""Streamlit demo: ECG heartbeat anomaly detection with LSTM and Bidirectional LSTM."""

import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

BASE = Path(__file__).parent

DATA_X_PATH = BASE / "demo_data" / "X_test.npy"
DATA_Y_PATH = BASE / "demo_data" / "y_test_multi.npy"
SCALER_PATH = BASE / "scaler" / "scaler.pkl"
MODEL_PATHS = {
    "Bidirectional LSTM": BASE / "checkpoints" / "best_model.keras",
    "LSTM": BASE / "checkpoints" / "plain_lstm" / "best_model.keras",
}
IMAGE_DIR = BASE / "images"

BEAT_TYPES = {
    1: "Normal",
    2: "R-on-T premature ventricular contraction",
    3: "Premature ventricular contraction",
    4: "Supraventricular premature or ectopic beat",
    5: "Unclassifiable beat",
}

RESULTS = [
    {"Metric": "Accuracy", "LSTM": "0.9622", "Bidirectional LSTM": "0.9729"},
    {"Metric": "Recall (abnormal)", "LSTM": "0.9119", "Bidirectional LSTM": "0.9365"},
    {"Metric": "Precision (abnormal)", "LSTM": "0.9971", "Bidirectional LSTM": "0.9983"},
    {"Metric": "F1 score", "LSTM": "0.9526", "Bidirectional LSTM": "0.9664"},
    {"Metric": "ROC AUC", "LSTM": "0.9957", "Bidirectional LSTM": "0.9968"},
    {"Metric": "Missed abnormal beats", "LSTM": "165", "Bidirectional LSTM": "119"},
    {"Metric": "False alarms", "LSTM": "5", "Bidirectional LSTM": "3"},
]

st.set_page_config(page_title="ECG Heartbeat Anomaly Detection", layout="wide")


# Loading helpers

def require_files():
    required = [DATA_X_PATH, DATA_Y_PATH, SCALER_PATH, *MODEL_PATHS.values()]
    missing = [str(p.relative_to(BASE)) for p in required if not p.exists()]
    if missing:
        st.error("Required files are missing:\n\n" + "\n".join(f"- {m}" for m in missing))
        st.stop()


@st.cache_resource
def load_models():
    return {name: load_model(path, compile=False) for name, path in MODEL_PATHS.items()}


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_data
def load_data():
    X = np.load(DATA_X_PATH)
    y_multi = np.load(DATA_Y_PATH)
    y_binary = (y_multi != 1).astype(int)
    mean_normal = X[y_binary == 0].mean(axis=0)
    return X, y_multi, y_binary, mean_normal


@st.cache_data
def get_probabilities(idx):
    """Abnormal probability from each model for one test beat."""
    X, _, _, _ = load_data()
    scaler = load_scaler()
    beat = scaler.transform(X[idx].reshape(1, -1)).reshape(1, 140, 1)
    return {
        name: float(model.predict(beat, verbose=0)[0, 0])
        for name, model in load_models().items()
    }


@st.cache_data
def image_to_base64(path_str):
    return base64.b64encode(Path(path_str).read_bytes()).decode()


def show_image(path, caption):
    """Embed the image directly in the page so it does not depend on the media endpoint."""
    data = image_to_base64(str(path))
    st.markdown(
        f'<img src="data:image/png;base64,{data}" style="width:100%;" alt="{caption}">',
        unsafe_allow_html=True,
    )
    st.caption(caption)


def beat_chart(beat, mean_normal, true_binary):
    """Native Streamlit line chart, drawn in the browser."""
    df = pd.DataFrame({"Average normal beat": mean_normal, "Selected beat": beat})
    df.index.name = "Time step"
    selected_color = "#d62728" if true_binary == 1 else "#1f77b4"
    st.line_chart(df, color=["#999999", selected_color], height=360)


require_files()
X, y_multi, y_binary, mean_normal = load_data()
n_beats = len(X)

# Header

st.title("ECG Heartbeat Anomaly Detection")
st.caption(
    "Research demo on the public ECG5000 benchmark. "
    "Not a medical device and not for diagnosis or clinical use."
)

tab_demo, tab_results, tab_about = st.tabs(["Demo", "Results", "About"])

# Demo tab

with tab_demo:
    st.sidebar.header("Controls")
    mode = st.sidebar.radio(
        "Beat selection",
        ["Any beat", "Normal beat", "Abnormal beat", "Specific index"],
    )

    if mode == "Specific index":
        idx = int(st.sidebar.number_input("Test beat index", 0, n_beats - 1, 0, step=1))
    else:
        if mode == "Normal beat":
            pool = np.where(y_binary == 0)[0]
        elif mode == "Abnormal beat":
            pool = np.where(y_binary == 1)[0]
        else:
            pool = np.arange(n_beats)

        draw = st.sidebar.button("Draw a new beat")
        if draw or st.session_state.get("mode") != mode or "idx" not in st.session_state:
            st.session_state["idx"] = int(np.random.default_rng().choice(pool))
            st.session_state["mode"] = mode
        idx = st.session_state["idx"]

    threshold = st.sidebar.slider(
        "Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05
    )
    st.sidebar.caption(
        "A beat is called abnormal when the model's abnormal probability "
        "is above this threshold. Lowering it catches more abnormal beats "
        "but raises the number of false alarms."
    )

    beat = X[idx]
    true_binary = int(y_binary[idx])
    true_type = BEAT_TYPES[int(y_multi[idx])]

    st.subheader(f"Test beat {idx}")
    if true_binary == 0:
        st.markdown("**True label:** Normal")
    else:
        st.markdown(f"**True label:** Abnormal ({true_type})")

    beat_chart(beat, mean_normal, true_binary)

    st.subheader("Model predictions")
    probs = get_probabilities(idx)
    columns = st.columns(len(probs))
    predicted_labels = []

    for col, (name, prob) in zip(columns, probs.items()):
        predicted = int(prob > threshold)
        predicted_labels.append(predicted)
        with col:
            st.markdown(f"#### {name}")
            st.markdown(f"**Prediction:** {'Abnormal' if predicted else 'Normal'}")
            st.metric("Abnormal probability", f"{prob:.3f}")
            st.progress(min(max(prob, 0.0), 1.0))
            if predicted == true_binary:
                st.success("Matches the true label")
            else:
                st.error("Does not match the true label")

    if len(set(predicted_labels)) > 1:
        st.info("The two models disagree on this beat.")

# Results tab

with tab_results:
    st.subheader("Test set performance")
    st.caption(
        "Measured on the 4,500 heartbeats of the official ECG5000 test file, "
        "with a decision threshold of 0.5."
    )
    st.table(RESULTS)

    st.subheader("Figures")
    figures = [
        ("Bidirectional LSTM training curves", "training_curves.png"),
        ("Bidirectional LSTM ROC curve", "roc_auc_curve.png"),
        ("LSTM training curves", "training_curves_plain_lstm.png"),
        ("LSTM ROC curve", "roc_auc_curve_plain_lstm.png"),
    ]
    left, right = st.columns(2)
    for i, (caption, filename) in enumerate(figures):
        path = IMAGE_DIR / filename
        target = left if i % 2 == 0 else right
        with target:
            if path.exists():
                show_image(path, caption)
            else:
                st.caption(f"Image not found: images/{filename}")

# About tab

with tab_about:
    st.subheader("Dataset")
    st.write(
        "ECG5000 comes from the UCR Time Series Classification Archive. Each sample is "
        "one heartbeat recorded as 140 consecutive values. The official split has 500 "
        "training and 4,500 test heartbeats, and this app browses the test heartbeats."
    )
    st.write(
        "The original data has five classes. This project merges the four abnormal "
        "types into one abnormal class, so the task is binary: normal versus abnormal."
    )

    st.subheader("Models")
    st.write(
        "Both models use one recurrent layer with 64 units, dropout of 0.3, and a single "
        "sigmoid output. One uses a standard LSTM and the other a Bidirectional LSTM, "
        "which reads each heartbeat forward and backward. Inputs are standardized with a "
        "scaler fit on the training data only."
    )

    st.subheader("Limitations")
    st.write(
        "Results come from a single training run per model, on one public dataset, with "
        "a small training set. The models do not identify the type of abnormality. "
        "This demo is for learning and portfolio purposes only."
    )