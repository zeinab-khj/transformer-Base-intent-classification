import optuna
import torch

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from datasets import load_from_disk

from metrics import compute_metrics

from config import (
    MODEL_NAME,
    TRAIN_DATA_PATH,
    VALID_DATA_PATH,
    NUM_LABELS,
    OUTPUT_DIR,
    BATCH_SIZE,
    NUM_EPOCHS
)


# Load datasets

train_dataset = load_from_disk(TRAIN_DATA_PATH)
valid_dataset = load_from_disk(VALID_DATA_PATH)



def objective(trial):

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-6,
        5e-5,
        log=True
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        0.0,
        0.1
    )


    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )


    training_args = TrainingArguments(

        output_dir=f"{OUTPUT_DIR}/trial_{trial.number}",

        learning_rate=learning_rate,

        weight_decay=weight_decay,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        evaluation_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",

        greater_is_better=True,

        logging_strategy="epoch",

        report_to="none",

        fp16=torch.cuda.is_available()
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )


    trainer.train()

    result = trainer.evaluate()


    return result["eval_macro_f1"]



if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=10
    )


    print("Best Parameters:")
    print(study.best_trial.params)

    print(
        f"Best Macro F1: {study.best_value:.4f}"
    )
