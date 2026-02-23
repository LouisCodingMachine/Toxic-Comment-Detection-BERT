import torch
from transformers import BertTokenizer
from model_utils import load_model

MODEL_PATH = "models/final_model"


def predict(text):
    # train.py에서 model과 함께 같은 경로에 저장된 토크나이저를 로드
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = load_model(MODEL_PATH)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        result = torch.argmax(probabilities).item()

    return result, probabilities.tolist()[0]


if __name__ == "__main__":
    sample_text = "이 제품 왜 이렇게 별로지"
    label, probs = predict(sample_text)
    print(f"결과: {'긍정' if label == 0 else '부정'} (확률: {probs})")
