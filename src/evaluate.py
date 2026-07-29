import json
import numpy as np
import pandas as pd

from pathlib import Path

from datasets import load_from_disk

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from config import (
    MODEL_DIR,
    VALID_DATA_PATH,
    REPORT_DIR
)


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------
    # Load model/tokenizer
    # -----------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )


    # -----------------------
    # Load validation dataset
    # -----------------------

    valid_dataset = load_from_disk(
        VALID_DATA_PATH
    )


    # -----------------------
    # Data Collator
    # -----------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )


    # -----------------------
    # Trainer
    # -----------------------

    trainer = Trainer(
        model=model,
        data_collator=data_collator
    )


    # -----------------------
    # Prediction
    # -----------------------

    predictions = trainer.predict(
        valid_dataset
    )

    logits = predictions.predictions

    labels = predictions.label_ids

    predicted_labels = np.argmax(
        logits,
        axis=1
    )


    # -----------------------
    # Metrics
    # -----------------------

    metrics = {

        "accuracy": accuracy_score(
            labels,
            predicted_labels
        ),

        "macro_precision": precision_score(
            labels,
            predicted_labels,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            labels,
            predicted_labels,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            labels,
            predicted_labels,
            average="macro"
        ),

        "weighted_f1": f1_score(
            labels,
            predicted_labels,
            average="weighted"
        )
    }


    with open(
        REPORT_DIR / "metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    # -----------------------
    # Classification Report
    # -----------------------

    report = classification_report(
        labels,
        predicted_labels,
        output_dict=True,
        zero_division=0
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    report_df.to_csv(
        REPORT_DIR / "classification_report.csv"
    )


    # -----------------------
    # Predictions dataframe
    # -----------------------

    df = valid_dataset.to_pandas()


    df["true_label"] = labels

    df["predicted_label"] = predicted_labels

    
    label_mapping = (
        df[
            ["true_label", "label_text"]
        ]
        .drop_duplicates()
        .set_index("true_label")["label_text"]
        .to_dict())

    df["true_class"] = df["true_label"].map(label_mapping)

    df["predicted_class"] = df["predicted_label"].map(label_mapping)

    df.to_csv(
        REPORT_DIR / "predictions.csv",
        index=False
    )


    # -----------------------
    # Misclassified samples
    # -----------------------

    errors = df[
        df["true_label"] != df["predicted_label"]
    ]


    errors.to_csv(
        REPORT_DIR / "misclassified_samples.csv",
        index=False
    )


    # -----------------------
    # Confusion Matrix
    # -----------------------

    cm = confusion_matrix(
        labels,
        predicted_labels
    )


    np.save(
        REPORT_DIR / "confusion_matrix.npy",
        cm
    )


    print("Evaluation completed.")
    print(metrics)



if __name__ == "__main__":
    main()
