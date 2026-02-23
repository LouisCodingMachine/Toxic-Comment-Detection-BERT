import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer


def load_and_split_data(file_path: str) -> DatasetDict:
    """CSV 파일을 로드하고 train / valid / test로 분할한다.
    분할 비율: Train 50% · Test 40% · Validation 10% """
    df = pd.read_csv(file_path)
    full_dataset = Dataset.from_pandas(df)

    # Train : Valid : Test = 5:4:1
    train_testvalid = full_dataset.train_test_split(test_size=0.5, seed=42)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.2, seed=42)

    return DatasetDict({
        "train": train_testvalid["train"],
        "test": test_valid["train"],
        "valid": test_valid["test"],
    })


def tokenize_datasets(ds_dict: DatasetDict, tokenizer: BertTokenizer) -> DatasetDict:
    """DatasetDict의 각 split에 토크나이저를 적용한다."""
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    return ds_dict.map(tokenize_func, batched=True)
