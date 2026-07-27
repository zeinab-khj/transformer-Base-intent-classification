import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer
)

from datasets import load_from_disk

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import pandas as pd

from config import (
    MODEL_DIR,
    VALID_DATA_PATH,
    REPORT_DIR
)


# -------------------------
# Load model and tokenizer
# -------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)


# -------------------------
# Load validation dataset
# -------------------------

valid_dataset = load_from_disk(
    VALID_DATA_PATH
)


# -------------------------
# Prediction
# -------------------------

trainer = Trainer(
    model=model
)


predictions = trainer.predict(
    valid_dataset
)


logits = predictions.predictions

labels = predictions.label_ids


predicted_labels = np.argmax(
    logits,
    axis=1
)


# -------------------------
# Metrics
# -------------------------

accuracy = accuracy_score(
    labels,
    predicted_labels
)

macro_f1 = f1_score(
    labels,
    predicted_labels,
    average="macro"
)

weighted_f1 = f1_score(
    labels,
    predicted_labels,
    average="weighted"
)


print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")


# -------------------------
# Classification Report
# -------------------------

report = classification_report(
    labels,
    predicted_labels,
    output_dict=True
)


report_df = pd.DataFrame(
    report
).transpose()


report_df.to_csv(
    REPORT_DIR / "classification_report.csv"
)


# -------------------------
# Confusion Matrix
# -------------------------

cm = confusion_matrix(
    labels,
    predicted_labels
)


np.save(
    REPORT_DIR / "confusion_matrix.npy",
    cm
)
