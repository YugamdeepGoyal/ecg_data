"""Export the raw ECG5000 test beats so the Streamlit app does not need Kaggle.

Run once from the project folder, with the same environment as the notebooks:

    python export_demo_data.py

Creates:
    demo_data/X_test.npy         raw (unscaled) test beats, shape (4500, 140)
    demo_data/y_test_multi.npy   original labels 1 to 5, shape (4500,)
"""

import os

import numpy as np

from src.load_data import download_data, load_file

path = download_data()
X_test, y_test = load_file(path, "ECG5000_TEST.ts")

os.makedirs("demo_data", exist_ok=True)
np.save("demo_data/X_test.npy", X_test)
np.save("demo_data/y_test_multi.npy", y_test.astype("int8"))

print("Saved X_test:", X_test.shape)
print("Saved y_test_multi:", y_test.shape, "labels:", np.unique(y_test))
