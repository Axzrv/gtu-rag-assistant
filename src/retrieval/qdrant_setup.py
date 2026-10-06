"""
Модуль подключения к Qdrant и создания гибридной коллекции.
"""
from qdrant_client import QdrantClient
from qdrant_client import models
import logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "gtu_docs"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 1024  # размерность deepvk/USER-bge-m3


def get_client() -> QdrantClient:
    """Создаёт подключение к Qdr ant."""
    return QdrantClient(
        host="localhost",
        port=6333,
        grpc_port=6334,
        prefer_grpc=True,
    )


def create_hybrid_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Создаёт гибридную коллекцию (dense + sparse) для RAG.

    :param recreate: если True — удаляет существующую коллекцию и создаёт заново.
    """
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        if recreate:
            logger.warning(f"Удаляю существующую коллекцию {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)
        else:
            logger.info(f"Коллекция {COLLECTION_NAME} уже существует")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )

    # Индексы для быстрой фильтрации по метаданным
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="page",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

    logger.info(f"Коллекция {COLLECTION_NAME} создана")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = get_client()
    create_hybrid_collection(client, recreate=True)
    info = client.get_collection(COLLECTION_NAME)
    print(f"Коллекция: {COLLECTION_NAME}")
    print(f"Точек (chunks): {info.points_count}")
    print(f"Конфигурация: {info.config.params}")