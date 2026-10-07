"""
Оценка RAG-системы через Ragas с локальной LLM-судьёй (LM Studio) и локальными эмбеддингами.
"""
# --- Обходной путь для совместимости ragas и langchain-community >= 0.4 ---
import sys
import types

_VERTEXAI_MODULE = "langchain_community.chat_models.vertexai"
if _VERTEXAI_MODULE not in sys.modules:
    _stub = types.ModuleType(_VERTEXAI_MODULE)

    class ChatVertexAI:
        pass

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules[_VERTEXAI_MODULE] = _stub
# --- Конец обходного пути ---

import json
import logging

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from src.generation.rag_chain import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Конфигурация LM Studio ---
LM_STUDIO_BASE_URL = "http://192.168.31.41:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = "qwen2.5-7b-instruct-1m"

# --- LLM-судья через LM Studio ---
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        model=LM_STUDIO_MODEL,
        temperature=0.0,
    )
)

# --- Локальные эмбеддинги (та же модель, что использовалась для индексации) ---
_embeddings_raw = HuggingFaceEmbeddings(
    model_name="deepvk/USER-bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
evaluator_embeddings = LangchainEmbeddingsWrapper(_embeddings_raw)


def load_golden_set(path: str = "eval/golden_set.json") -> list[dict]:
    """Загружает 'золотой набор' вопросов."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(golden: list[dict]) -> Dataset:
    """Прогоняет 'золотой набор' через RAG-систему и формирует Dataset для Ragas."""
    samples = []
    for item in golden:
        logger.info(f"Прогон: {item['question'][:60]}...")
        result = generate_answer(item["question"], top_k=5)
        samples.append({
            "user_input": item["question"],
            "response": result["answer"],
            "retrieved_contexts": [c["text"] for c in result["chunks"]],
            "reference": item["ground_truth"],
        })
    return Dataset.from_list(samples)


def main():
    golden = load_golden_set()
    logger.info(f"Загружено {len(golden)} вопросов")

    dataset = build_dataset(golden)

    # RunConfig для локальной LLM-судьи: медленнее и менее параллельна, чем OpenAI
    run_config = RunConfig(
        timeout=600,
        max_retries=3,
        max_workers=2,
    )

    logger.info("Запуск оценки Ragas...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()
    df.to_csv("eval/ragas_results.csv", index=False, encoding="utf-8")

    print("\n=== Средние метрики ===")
    print(result)
    print("\nДетальные результаты: eval/ragas_results.csv")


if __name__ == "__main__":
    main()