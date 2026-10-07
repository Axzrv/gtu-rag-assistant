"""
Гибридный поиск: dense (USER-bge-m3) + sparse (BM25) + реранкинг через кросс-энкодер.
"""
import logging

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
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

# --- Определяем устройство один раз ---
_device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Модели загружаем один раз на уровне модуля (чтобы не перезагружать при каждом вызове) ---
_dense_model = SentenceTransformer("deepvk/USER-bge-m3", device=_device)
_sparse_model = SparseTextEmbedding(
    model_name="Qdrant/bm25",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    if _device == "cuda" else ["CPUExecutionProvider"],
)
_reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device=_device,
    max_length=512,
)


def _rrf_fusion(
    dense_results: list,
    sparse_results: list,
    weights: tuple[float, float] = (0.4, 0.6),
    k: int = 60,
) -> dict:
    """
    Reciprocal Rank Fusion — объединяет два ранжированных списка.

    :param dense_results: список результатов dense-поиска (по убыванию score)
    :param sparse_results: список результатов sparse-поиска (по убыванию score)
    :param weights: (вес_dense, вес_sparse). Например, (0.4, 0.6) даёт больше веса BM25.
    :param k: константа RRF (стандарт = 60).
    :return: словарь {point_id: fused_score}
    """
    w_dense, w_sparse = weights
    scores: dict = {}

    for rank, point in enumerate(dense_results, start=1):
        scores[point.id] = scores.get(point.id, 0) + w_dense / (k + rank)

    for rank, point in enumerate(sparse_results, start=1):
        scores[point.id] = scores.get(point.id, 0) + w_sparse / (k + rank)

    return scores


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Переранжирует кандидатов с помощью кросс-энкодера.

    :param query: поисковый запрос
    :param candidates: список словарей с полями text, source, page, score
    :param top_k: сколько оставить после реранкинга
    :return: отсортированный список словарей
    """
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    scores = _reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


def hybrid_search(
    query: str,
    top_k: int = 5,
    prefetch_multiplier: int = 4,
    use_reranker: bool = True,
) -> list[dict]:
    """
    Гибридный поиск: dense + sparse через RRF, затем опциональный реранкинг.

    :param query: поисковый запрос
    :param top_k: сколько результатов вернуть в итоге
    :param prefetch_multiplier: во сколько раз больше кандидатов взять из Qdrant
                                (для последующего реранкинга)
    :param use_reranker: если True — переранжировать через кросс-энкодер
    :return: список словарей с полями text, source, page, score (и rerank_score)
    """
    # --- 1. Dense-вектор запроса ---
    query_dense = _dense_model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    # --- 2. Sparse BM25-вектор запроса ---
    sparse_result = list(_sparse_model.embed([query]))[0]
    query_sparse = models.SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    # --- 3. Берём больше кандидатов из Qdrant (для реранкинга) ---
    candidate_limit = top_k * prefetch_multiplier
    client = get_client()

    # Dense-поиск отдельно (чтобы потом объединить вручную)
    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_dense,
        using=DENSE_VECTOR_NAME,
        limit=candidate_limit,
        with_payload=True,
    ).points

    # Sparse-поиск отдельно
    sparse_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_sparse,
        using=SPARSE_VECTOR_NAME,
        limit=candidate_limit,
        with_payload=True,
    ).points

    # --- 4. RRF-объединение двух списков ---
    # Собираем все уникальные точки по id
    all_points: dict = {}
    for p in dense_results:
        all_points[p.id] = p
    for p in sparse_results:
        all_points[p.id] = p

    rrf_scores = _rrf_fusion(
        dense_results,
        sparse_results,
        weights=(0.4, 0.6),  # dense=0.4, sparse=0.6 (больше веса BM25)
        k=60,
    )

    # Формируем объединённый список кандидатов, отсортированный по fused score
    candidates = []
    for point_id, fused_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        p = all_points[point_id]
        candidates.append({
            "text": p.payload["text"],
            "source": p.payload["source"],
            "page": p.payload["page"],
            "score": float(fused_score),
        })

    # --- 5. Реранкинг (опционально) ---
    if use_reranker and len(candidates) > top_k:
        return rerank(query, candidates, top_k=top_k)

    return candidates[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Какие параметры вибрации считаются критическими для газовой турбины?",
        "Требования ГОСТ 34365-2017 к системе охлаждения",
        "Ресурс до капитального ремонта турбины V64.3A",
        "Что такое схема охлаждения лопаток?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"ЗАПРОС: {q}")
        print('=' * 70)

        # С реранкингом
        results = hybrid_search(q, top_k=3, use_reranker=True)
        for i, r in enumerate(results, 1):
            print(f"\n--- Результат {i} ---")
            print(f"RRF score: {r['score']:.4f} | Rerank score: {r.get('rerank_score', 'N/A')}")
            print(f"Источник: {r['source']}, стр. {r['page']}")
            print(f"Текст: {r['text'][:250]}...")