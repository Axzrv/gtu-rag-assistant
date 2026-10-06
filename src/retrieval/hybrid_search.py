"""
Гибридный поиск: dense (USER-bge-m3) + sparse (BM25) с объединением через RRF.
"""
import logging

import torch
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import models

from src.retrieval.qdrant_setup import (
    get_client,
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные модели — загружаем один раз
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dense_model = SentenceTransformer("deepvk/USER-bge-m3", device=_device)
_sparse_model = SparseTextEmbedding(
    model_name="Qdrant/bm25",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    if _device == "cuda" else ["CPUExecutionProvider"],
)

def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Гибридный поиск: dense + sparse через RRF.

    :param query: поисковый запрос
    :param top_k: количество результатов
    :return: список словарей с полями text, source, page, score
    """
    # Dense-вектор запроса
    query_dense = _dense_model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    # Sparse-вектор запроса (BM25)
    sparse_result = list(_sparse_model.embed([query]))[0]
    query_sparse = models.SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    # Гибридный поиск через prefetch + RRF
    client = get_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=query_dense,
                using=DENSE_VECTOR_NAME,
                limit=top_k * 2,
            ),
            models.Prefetch(
                query=query_sparse,
                using=SPARSE_VECTOR_NAME,
                limit=top_k * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "page": r.payload["page"],
            "score": r.score,
        }
        for r in results
    ]

if __name__ == "__main__":
    test_queries = [
        "Какие параметры вибрации считаются критическими для газовой турбины?",
        "Требования ГОСТ 34365-2017 к системе охлаждения",
        "Ресурс до капитального ремонта турбины V64.3A",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"ЗАПРОС: {q}")
        print('=' * 70)
        results = hybrid_search(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n--- Результат {i} (score={r['score']:.4f}) ---")
            print(f"Источник: {r['source']}, стр. {r['page']}")
            print(f"Текст: {r['text'][:250]}...")