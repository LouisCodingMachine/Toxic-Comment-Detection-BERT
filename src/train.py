import numpy as np
import evaluate
from transformers import BertTokenizer, Trainer, TrainingArguments
from data_loader import load_and_split_data, tokenize_datasets
from model_utils import build_model

TOKENIZER_NAME = "bert-base-multilingual-cased"  # 어휘 사전(토크나이저)용
MODEL_SAVE_PATH = "models/final_model"


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train():
    raw_datasets = load_and_split_data("data/raw/hate_comments.csv")

    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer)

    model = build_model(num_labels=2)

    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 추론 시 같은 경로에서 토크나이저도 로드할 수 있도록 함께 저장
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)


if __name__ == "__main__":
    train()
