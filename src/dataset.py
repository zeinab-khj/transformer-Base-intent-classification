from datasets import Dataset
from datasets import ClassLabel
from config import TRAIN_DATA_PATH, VALID_DATA_PATH
    

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

train_dataset = dataset_split["train"]
valid_dataset = dataset_split["validation"]

train_dataset.save_to_disk(TRAIN_DATA_PATH)

valid_dataset.save_to_disk(VALID_DATA_PATH)
