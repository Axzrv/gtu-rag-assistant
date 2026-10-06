"""
Загрузка чанков в Qdrant: dense-эмбеддинги (USER-bge-m3) + sparse BM25 (Qdrant/bm25).
"""
import json
import logging
import torch
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import models

from src.retrieval.qdrant_setup import get_client, COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Размер батча для upsert (загрузки в Qdrant)
BATCH_SIZE = 64
# Размер батча для кодирования (на CPU меньше = меньше памяти)
ENCODE_BATCH = 32 if torch.cuda.is_available() else 8

def load_chunks(path: str) -> list[dict]:
    """Загружает чанки из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def index_chunks(chunks: list[dict], batch_size: int = BATCH_SIZE) -> None:
    """Генерирует эмбеддинги и загружает чанки в Qdrant."""
    logger.info("Загрузка моделей...")

    # Определяем устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Использую устройство: {device}")

    # Dense: эмбеддинги на GPU
    dense_model = SentenceTransformer("deepvk/USER-bge-m3", device=device)

    # Sparse: BM25 с поддержкой GPU
    sparse_model = SparseTextEmbedding(
        model_name="Qdrant/bm25",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda" else ["CPUExecutionProvider"]
    )

    logger.info("Модели загружены. Начинаю кодирование...")

    client = get_client()
    points_buffer = []
    total = len(chunks)

    for i, chunk in enumerate(tqdm(chunks, desc="Кодирование чанков")):
        text = chunk["text"]

        # Dense-эмбеддинг (нормализованный для косинусной близости)
        dense_vec = dense_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Sparse BM25-эмбеддинг
        sparse_result = list(sparse_model.embed([text]))[0]
        sparse_vec = models.SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )

        # Формируем точку для Qdrant
        point = models.PointStruct(
            id=i,
            vector={
                DENSE_VECTOR_NAME: dense_vec,
                SPARSE_VECTOR_NAME: sparse_vec,
            },
            payload={
                "text": text,
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
            },
        )
        points_buffer.append(point)

        # Загружаем батчами
        if len(points_buffer) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=points_buffer)
            points_buffer = []

    # Остаток
    if points_buffer:
        client.upsert(collection_name=COLLECTION_NAME, points=points_buffer)

    logger.info(f"Готово. Загружено {total} чанков.")

if __name__ == "__main__":
    chunks = load_chunks("data/processed/chunks.json")
    index_chunks(chunks)