"""bert_modeling.py 기반 BERT 모델 생성·저장·로딩 유틸리티."""

from transformers import BertConfig
from bert_modeling import BertForSequenceClassification

# bert-base-multilingual-cased 아키텍처와 동일한 기본 설정값.
# 사전 학습된 가중치 없이 동일한 구조를 로컬 코드로만 초기화할 때 사용한다.
BERT_BASE_CONFIG = {
    "vocab_size": 119547,          # bert-base-multilingual-cased 어휘 사전 크기
    "hidden_size": 768,            # 히든 레이어 차원 수
    "num_hidden_layers": 12,       # 트랜스포머 레이어 수
    "num_attention_heads": 12,     # 어텐션 헤드 수
    "intermediate_size": 3072,     # FFN 내부 차원 수 (hidden_size × 4)
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,          # token_type_ids 종류 수 (segment A / B)
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "pad_token_id": 0,
    "chunk_size_feed_forward": 0,
    "classifier_dropout": None,
}


def build_config(num_labels: int, **overrides) -> BertConfig:
    """BertConfig를 생성한다.

    Args:
        num_labels: 분류 레이블 수.
        **overrides: BERT_BASE_CONFIG의 기본값을 덮어쓸 설정.

    Returns:
        BertConfig 인스턴스.
    """
    params = {**BERT_BASE_CONFIG, "num_labels": num_labels}
    params.update(overrides)
    return BertConfig(**params)


def build_model(num_labels: int, **overrides) -> BertForSequenceClassification:
    """랜덤 초기화된 BertForSequenceClassification 모델을 생성한다.

    사전 학습된 가중치(from_pretrained) 없이, bert_modeling.py의 로컬 코드와
    BertConfig만으로 모델을 구성한다.

    Args:
        num_labels: 분류 레이블 수 (예: 이진분류 → 2).
        **overrides: BERT_BASE_CONFIG의 기본값을 덮어쓸 설정.

    Returns:
        BertForSequenceClassification 인스턴스.
    """
    config = build_config(num_labels, **overrides)
    return BertForSequenceClassification(config)


def load_model(model_path: str) -> BertForSequenceClassification:
    """로컬에 저장된 모델 체크포인트를 불러온다.

    학습 후 save_pretrained()로 저장된 config.json과 가중치 파일을
    bert_modeling.py의 BertForSequenceClassification으로 로드한다.

    Args:
        model_path: 모델이 저장된 로컬 디렉토리 경로.

    Returns:
        BertForSequenceClassification 인스턴스.
    """
    return BertForSequenceClassification.from_pretrained(model_path)
