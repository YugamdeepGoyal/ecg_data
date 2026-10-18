import kagglehub
import numpy as np
import os


def download_data():
    path = kagglehub.dataset_download("salsabilahmid/ecg50000")
    print("Path to dataset files:", path)
    return path


def load_file(path, filename):
    file_path = os.path.join(path, filename)
    X = []
    y = []
    start = False
    with open(file=file_path) as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("@data"):
                start = True
                continue
            if start and line:
                values, label = line.rsplit(":", 1)
                X.append([float(v) for v in values.split(",")])
                y.append(int(label))
    return np.array(X), np.array(y)
