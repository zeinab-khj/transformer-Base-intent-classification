from datasets import Dataset
from datasets import ClassLabel

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

dataset_split.save_to_disk(
    "/drive/MyDrive/dataset_split"
)
