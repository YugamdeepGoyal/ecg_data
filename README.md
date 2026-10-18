# ECG Heartbeat Anomaly Detection with LSTM and Bidirectional LSTM

Binary classification of heartbeats (normal versus abnormal) on the ECG5000 benchmark dataset, using recurrent neural networks built with TensorFlow and Keras. The project compares a plain LSTM against a Bidirectional LSTM under identical preprocessing, seeding, and training settings.

## Summary

| Model | Test Accuracy | Abnormal Recall | Abnormal Precision | F1 | ROC AUC |
| :- | :- | :- | :- | :- | :- |
| LSTM | 96.22% | 91.19% | 99.71% | 0.953 | 0.9957 |
| Bidirectional LSTM | 97.29% | 93.65% | 99.83% | 0.966 | 0.9968 |

All figures are measured on 4,500 heartbeats that the models never saw during training. The Bidirectional LSTM is the stronger model on every metric, and its main advantage is higher recall on abnormal beats (119 missed beats compared with 165 for the plain LSTM).

## Problem Statement

An electrocardiogram (ECG) records the electrical activity of the heart. Each heartbeat has a characteristic shape, and deviations from that shape can indicate arrhythmias. The goal of this project is to classify a single recorded heartbeat as normal or abnormal directly from its raw signal, without hand-crafted features.

This is a sequence classification task. The order of values within a heartbeat matters, which makes recurrent architectures a natural fit.

## Dataset

The project uses ECG5000, originally from the UCR Time Series Classification Archive. The data was obtained through Kaggle (`salsabilahmid/ecg50000`) and is downloaded automatically with `kagglehub`.

| Property | Value |
| :- | :- |
| Total heartbeats | 5,000 |
| Training file | 500 heartbeats |
| Test file | 4,500 heartbeats |
| Time steps per heartbeat | 140 |
| Features per time step | 1 |
| Original classes | 5 |

The original labels are:

| Label | Meaning |
| :- | :- |
| 1 | Normal |
| 2 | R-on-T premature ventricular contraction |
| 3 | Premature ventricular contraction |
| 4 | Supraventricular premature or ectopic beat |
| 5 | Unclassifiable beat |

For this project the task is binary. Label 1 is mapped to 0 (normal), and labels 2 to 5 are merged into 1 (abnormal).

| Split | Normal (0) | Abnormal (1) |
| :- | :- | :- |
| Train | 292 | 208 |
| Test | 2,627 | 1,873 |

The official UCR split of 500 training and 4,500 test samples is kept as is. This makes the results comparable with published work on the same benchmark.

## Method

### Preprocessing

1. Load the `.ts` files and parse each row into 140 signal values and a label.
2. Convert labels to binary.
3. Standardize the signal with `StandardScaler`. The scaler is fit on the training set only and then applied to the test set, which prevents information from the test set leaking into training.
4. Reshape the arrays to `(samples, 140, 1)`, the input shape expected by recurrent layers.

### Models

Both models share the same structure. The only difference is the recurrent layer.

| Layer | LSTM model | Bidirectional LSTM model |
| :- | :- | :- |
| Input | (140, 1) | (140, 1) |
| Recurrent | LSTM(64) | Bidirectional(LSTM(64)) |
| Regularization | Dropout(0.3) | Dropout(0.3) |
| Output | Dense(1, sigmoid) | Dense(1, sigmoid) |

A Bidirectional LSTM reads each heartbeat forward and backward and combines both views. This suits the task because each heartbeat is fully recorded before classification, so the later part of the signal can inform the interpretation of the earlier part.

### Training Configuration

| Setting | Value |
| :- | :- |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Maximum epochs | 100 |
| Validation split | 15% of the training data (75 heartbeats) |
| Random seed | 42, with TensorFlow deterministic operations enabled |
| EarlyStopping | Monitor validation loss, patience 15, restore best weights |
| ReduceLROnPlateau | Monitor validation loss, factor 0.5, patience 5, minimum learning rate 1e-5 |
| ModelCheckpoint | Save the best model by validation loss |
| TensorBoard | Separate log directory for each model |

## Results

### Test Set Performance

| Metric | LSTM | Bidirectional LSTM |
| :- | :- | :- |
| Accuracy | 0.9622 | 0.9729 |
| Recall (abnormal) | 0.9119 | 0.9365 |
| Precision (abnormal) | 0.9971 | 0.9983 |
| F1 score | 0.9526 | 0.9664 |
| ROC AUC | 0.9957 | 0.9968 |

### Confusion Matrices (Test Set)

Rows are actual classes and columns are predicted classes.

LSTM:

| | Predicted normal | Predicted abnormal |
| :- | :- | :- |
| Actual normal | 2,622 | 5 |
| Actual abnormal | 165 | 1,708 |

Bidirectional LSTM:

| | Predicted normal | Predicted abnormal |
| :- | :- | :- |
| Actual normal | 2,624 | 3 |
| Actual abnormal | 119 | 1,754 |

### Training Versus Test (Bidirectional LSTM)

| Metric | Train | Test |
| :- | :- | :- |
| Accuracy | 0.990 | 0.973 |
| Recall (abnormal) | 0.981 | 0.936 |
| ROC AUC | 0.999 | 0.997 |

The gap between training and test performance is small, which indicates that the model generalizes well.

### Figures

Bidirectional LSTM training curves (loss, accuracy, and learning rate schedule):

![Bidirectional LSTM training curves](images/training_curves.png)

Bidirectional LSTM ROC curve:

![Bidirectional LSTM ROC curve](images/roc_auc_curve.png)

Plain LSTM training curves:

![LSTM training curves](images/training_curves_plain_lstm.png)

Plain LSTM ROC curve:

![LSTM ROC curve](images/roc_auc_curve_plain_lstm.png)

## Discussion

1. Both models separate the two classes very well, with ROC AUC above 0.995. The difference in AUC between them is small (about 0.001).
2. The larger difference appears in recall at the default 0.5 threshold. The Bidirectional LSTM detects 46 more abnormal beats than the plain LSTM.
3. Both models are very precise. They rarely flag a normal beat as abnormal (3 and 5 false alarms out of 2,627 normal beats).
4. The main weakness of both models is missed abnormal beats. Because the AUC is high, this is largely a matter of the decision threshold rather than a lack of separability. In a screening setting, a lower threshold would trade a small amount of precision for higher recall.
5. Early stopping selected the best validation epoch, and the learning rate schedule reduced the rate in steps as validation loss plateaued.

## Limitations

- Results come from a single run per model with seed 42. Differences of about one percentage point between models could partly be due to random variation. Averaging over several seeds would give a more reliable comparison.
- The validation set contains only 75 heartbeats, so validation curves are noisy and should be interpreted cautiously.
- Only one dataset was used. Performance on signals from other devices, patients, or recording conditions is unknown.
- The binary formulation merges four abnormal beat types, so the model does not identify which type of abnormality is present.
- This project is a machine learning exercise on a public benchmark. It is not a medical device and must not be used for diagnosis or clinical decisions.

## Project Structure

```
.
├── src/
│   └── load_data.py             Dataset download and parsing functions
├── model_1.ipynb                Bidirectional LSTM experiment
├── model_plain_lstm.ipynb       Plain LSTM experiment
├── images/                      Saved figures (training curves, ROC curves)
├── checkpoints/                 Saved best model weights
├── logs/                        TensorBoard logs
├── scaler/                      Fitted StandardScaler files
└── README.md
```

## Setup

Requirements:

- Python 3.12
- TensorFlow and Keras
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- joblib
- kagglehub

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv tf_env
source tf_env/bin/activate
pip install tensorflow numpy scikit-learn matplotlib seaborn joblib kagglehub
```

## Usage

1. Open `model_1.ipynb` to run the Bidirectional LSTM experiment, or `model_plain_lstm.ipynb` for the plain LSTM experiment.
2. Run the cells from top to bottom. The dataset is downloaded automatically on the first run through `kagglehub`.
3. Figures are saved to `images/`, the best model to `checkpoints/`, and the fitted scaler to `scaler/`.

To view training curves interactively, launch TensorBoard from the project folder and point it at the `logs` directory.

## Reproducibility

The notebooks set the Python, NumPy, and TensorFlow random seeds to 42 and enable deterministic TensorFlow operations. Restart the kernel and run all cells in order to reproduce the reported numbers. Exact values may differ across hardware, operating systems, and TensorFlow versions.

## Possible Extensions

- Compare against a one-dimensional convolutional network and a classical baseline such as logistic regression.
- Repeat training over multiple seeds and report the mean and standard deviation.
- Select the decision threshold on a validation set to improve recall on abnormal beats.
- Extend to the original five-class problem with class weighting for the rare classes.
- Add an interactive demo application for classifying a selected heartbeat.

## Acknowledgments

- ECG5000 dataset from the UCR Time Series Classification Archive.
- Kaggle dataset mirror: `salsabilahmid/ecg50000`.
