import torch

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer
)



from config import (
    MODEL_NAME,
    TRAIN_DATA_PATH,
    VALID_DATA_PATH,
    NUM_LABELS,
    MODEL_DIR,
    BATCH_SIZE,
    NUM_EPOCHS
)

from metrics import compute_metrics
from datasets import load_from_disk



# -------------------------
# Load datasets
# -------------------------

train_dataset = load_from_disk(TRAIN_DATA_PATH)

valid_dataset = load_from_disk(VALID_DATA_PATH)



# -------------------------
# Load final model
# -------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)



# -------------------------
# Best hyperparameters
# -------------------------

learning_rate = 3.5498328479599875e-05

weight_decay = 0.09803257513396252



# -------------------------
# Training configuration
# -------------------------

training_args = TrainingArguments(

    output_dir=MODEL_DIR,

    learning_rate=learning_rate,

    weight_decay=weight_decay,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    logging_strategy="epoch",

    report_to="none",

    fp16=torch.cuda.is_available()
)



# -------------------------
# Trainer
# -------------------------

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train_dataset,

    eval_dataset=tokenized_valid_dataset,

    compute_metrics=compute_metrics,
    data_collator=data_collator
)



# -------------------------
# Train
# -------------------------

trainer.train()



# -------------------------
# Save final model
# -------------------------

trainer.save_model(
    MODEL_DIR
)


print(
    f"Model saved to {MODEL_DIR}"
)
