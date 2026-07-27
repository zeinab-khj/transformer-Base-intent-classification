from datasets import Dataset
from datasets import ClassLabel
from transformers import AutoTokenizer
from config import TRAIN_DATA_PATH, VALID_DATA_PATH, MODEL_NAME
    

train_dataset = Dataset.from_pandas(train_df)


num_labels = train_df['label'].nunique()

train_dataset = train_dataset.cast_column(
    "label", ClassLabel(num_classes=num_labels)
)

dataset_split = train_dataset.train_test_split(
    test_size=0.1,
    seed=42,
    stratify_by_column="label"
)

dataset_split["validation"] = dataset_split["test"]
del dataset_split["test"]

train = dataset_split["train"]
valid = dataset_split["validation"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

train_dataset = train.map(tokenize_function, batched=True)
valid_dataset = valid.map(tokenize_function, batched=True)

train_dataset.save_to_disk(TRAIN_DATA_PATH)

valid_dataset.save_to_disk(VALID_DATA_PATH)
