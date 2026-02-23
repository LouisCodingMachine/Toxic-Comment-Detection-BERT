import evaluate
import numpy as np
from transformers import BertTokenizer, Trainer
from data_loader import load_and_split_data, tokenize_datasets
from model_utils import load_model

MODEL_PATH = "models/final_model"


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def evaluate_on_test():
    raw_datasets = load_and_split_data("data/raw/hate_comments.csv")

    # 학습 시 함께 저장된 토크나이저를 로드하여 동일한 토크나이징 보장
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer)

    model = load_model(MODEL_PATH)

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    # 데이터 로드 (학습 때와 동일한 시드 사용)
    results = trainer.evaluate(tokenized_datasets["test"])

    print("-" * 30)
    print(f"✅ 최종 테스트 정확도(Accuracy): {results.get('eval_accuracy'):.4f}")
    print(f"📉 최종 테스트 손실(Loss): {results.get('eval_loss'):.4f}")
    print("-" * 30)
    print("상세 결과:", results)


if __name__ == "__main__":
    evaluate_on_test()
