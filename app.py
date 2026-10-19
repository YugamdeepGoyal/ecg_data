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
GITHUB_URL = "https://github.com/YugamdeepGoyal/ecg_data"

BEAT_TYPES = {
    1: "Normal",
    2: "R-on-T premature ventricular contraction",
    3: "Premature ventricular contraction",
    4: "Supraventricular premature or ectopic beat",
    5: "Unclassifiable beat",
}

BEAT_CHOICES = ["Any beat", "Normal beat", "Abnormal beat"]

RESULTS = [
    {"Metric": "Accuracy", "LSTM": "0.9622", "Bidirectional LSTM": "0.9729"},
    {"Metric": "Recall (abnormal)", "LSTM": "0.9119", "Bidirectional LSTM": "0.9365"},
    {"Metric": "Precision (abnormal)", "LSTM": "0.9971", "Bidirectional LSTM": "0.9983"},
    {"Metric": "F1 score", "LSTM": "0.9526", "Bidirectional LSTM": "0.9664"},
    {"Metric": "ROC AUC", "LSTM": "0.9957", "Bidirectional LSTM": "0.9968"},
    {"Metric": "Missed abnormal beats", "LSTM": "165", "Bidirectional LSTM": "119"},
    {"Metric": "False alarms", "LSTM": "5", "Bidirectional LSTM": "3"},
]

st.set_page_config(page_title="ECG Heartbeat Anomaly Detection", layout="centered")


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
    st.line_chart(df, color=["#999999", selected_color], height=320)


def draw_beat():
    """Pick a random test beat of the kind chosen by the visitor."""
    _, _, y_binary, _ = load_data()
    mode = st.session_state.get("mode", "Any beat")
    if mode == "Normal beat":
        pool = np.where(y_binary == 0)[0]
    elif mode == "Abnormal beat":
        pool = np.where(y_binary == 1)[0]
    else:
        pool = np.arange(len(y_binary))
    st.session_state["idx"] = int(np.random.default_rng().choice(pool))


require_files()
X, y_multi, y_binary, mean_normal = load_data()

if "idx" not in st.session_state:
    draw_beat()

# Header

st.title("ECG Heartbeat Anomaly Detection")
st.write(
    "Two neural networks, trained on thousands of recorded heartbeats, "
    "try to tell normal heartbeats from abnormal ones. Pick a heartbeat and see what they decide."
)
st.caption(
    "Research and portfolio project on a public benchmark dataset. "
    "Not a medical device and not for diagnosis."
)

tab_try, tab_perf, tab_about = st.tabs(["Try it", "Performance", "About"])

# Try it tab

with tab_try:
    st.radio(
        "Which kind of heartbeat?",
        BEAT_CHOICES,
        key="mode",
        horizontal=True,
        on_change=draw_beat,
    )
    st.button("Show me a heartbeat", on_click=draw_beat, type="primary")

    idx = st.session_state["idx"]
    beat = X[idx]
    true_binary = int(y_binary[idx])
    true_type = BEAT_TYPES[int(y_multi[idx])]

    st.subheader("The heartbeat")
    beat_chart(beat, mean_normal, true_binary)
    st.caption(
        "The colored line is the selected heartbeat, recorded as 140 measurements in a row. "
        "The gray line is the average normal heartbeat, for comparison."
    )

    if true_binary == 0:
        st.markdown("**What the dataset says:** Normal")
    else:
        st.markdown(f"**What the dataset says:** Abnormal, type: {true_type}")

    st.subheader("What the models say")
    probs = get_probabilities(idx)
    threshold = st.session_state.get("threshold", 0.5)

    columns = st.columns(len(probs))
    correctness = {}
    for col, (name, prob) in zip(columns, probs.items()):
        predicted = int(prob > threshold)
        correct = predicted == true_binary
        correctness[name] = correct
        with col:
            st.markdown(f"**{name}**")
            st.markdown(f"Verdict: **{'Abnormal' if predicted else 'Normal'}**")
            st.progress(
                min(max(prob, 0.0), 1.0),
                text=f"Chance of abnormal: {prob * 100:.0f}%",
            )
            if correct:
                st.success("Correct")
            else:
                st.error("Incorrect")

    n_correct = sum(correctness.values())
    if n_correct == len(correctness):
        st.info("Both models got this one right.")
    elif n_correct == 0:
        st.info("Both models got this one wrong. Some heartbeats are genuinely hard to tell apart.")
    else:
        right = [n for n, ok in correctness.items() if ok][0]
        st.info(f"The models disagree on this beat. Only the {right} was correct.")

    with st.expander("Advanced: decision threshold"):
        st.write(
            "A heartbeat is called abnormal when the model's chance of abnormal is above this "
            "threshold. Lowering it catches more abnormal beats but causes more false alarms. "
            "The default is 50%."
        )
        st.slider(
            "Decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key="threshold",
        )

# Performance tab

with tab_perf:
    st.write(
        "Both models were tested on 4,500 heartbeats they never saw during training. "
        "The Bidirectional LSTM, which reads each heartbeat forward and backward, "
        "performed better, mainly because it missed fewer abnormal beats."
    )
    st.table(RESULTS)

    st.subheader("Figures")
    figures = [
        ("Bidirectional LSTM training curves", "training_curves.png"),
        ("Bidirectional LSTM ROC curve", "roc_auc_curve.png"),
        ("LSTM training curves", "training_curves_plain_lstm.png"),
        ("LSTM ROC curve", "roc_auc_curve_plain_lstm.png"),
    ]
    for caption, filename in figures:
        path = IMAGE_DIR / filename
        if path.exists():
            show_image(path, caption)
        else:
            st.caption(f"Image not found: images/{filename}")

# About tab

with tab_about:
    st.subheader("What this is")
    st.write(
        "A machine learning project that classifies a single heartbeat as normal or abnormal "
        "directly from its recorded signal, with no hand-made features."
    )

    st.subheader("Data")
    st.write(
        "ECG5000, from the UCR Time Series Classification Archive. Each sample is one heartbeat "
        "recorded as 140 consecutive values. The original data has five classes. This project "
        "merges the four abnormal types into one, so the task is normal versus abnormal. "
        "This page shows heartbeats from the official test set."
    )

    st.subheader("Models")
    st.write(
        "Both models use one recurrent layer with 64 units, dropout of 0.3, and a single "
        "sigmoid output. One is a standard LSTM and the other is a Bidirectional LSTM. "
        "Signals are standardized with a scaler fit on the training data only."
    )

    st.subheader("Limitations")
    st.write(
        "Each model was trained once, on one public dataset with a small training set. "
        "The models do not say which type of abnormality a beat has, and they were not tested "
        "on recordings from other devices or patients. This demo is for learning and "
        "portfolio purposes only."
    )

    st.subheader("Source code")
    st.markdown(f"[View the project on GitHub]({GITHUB_URL})")