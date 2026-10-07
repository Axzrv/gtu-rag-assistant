"""
RAG-цепочка: гибридный поиск -> формирование промпта -> LLM (LM Studio) -> ответ с цитатами.
"""
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.hybrid_search import hybrid_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Конфигурация LM Studio ---
LM_STUDIO_BASE_URL = "http://192.168.31.41:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = "qwen2.5-7b-instruct-1m"

# --- Промпт-инструкция для LLM ---
RAG_PROMPT = """Ты — технический ассистент по парогазовым турбинам.

Используй предоставленный контекст для ответа. Если в контексте есть ЧАСТИЧНАЯ информация — приведи её и укажи, чего не хватает.
Отвечай строго на русском языке.

Правила:
1. Каждый факт подтверждай ссылкой: (Источник: имя_файла, стр. N).
2. Если информации нет совсем — скажи: "В предоставленных документах нет информации по этому вопросу."
3. Не выдумывай факты, которых нет в контексте.
4. Отвечай по существу, без повторов вопроса.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""

def format_context(chunks: list[dict]) -> str:
    """Форматирует найденные чанки в единый контекст для LLM."""
    if not chunks:
        return "Контекст пуст."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Фрагмент {i}]\n"
            f"Источник: {chunk['source']}, стр. {chunk['page']}\n"
            f"Текст: {chunk['text']}\n"
        )
    return "\n---\n".join(parts)


def get_llm() -> ChatOpenAI:
    """Создаёт подключение к LM Studio через OpenAI-совместимый API."""
    return ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        model=LM_STUDIO_MODEL,
        temperature=0.1,
        max_tokens=1024,
    )


def generate_answer(question: str, top_k: int = 5) -> dict:
    """
    Генерирует ответ на вопрос на основе RAG.

    :param question: вопрос пользователя
    :param top_k: количество чанков для контекста
    :return: словарь с answer, sources и chunks
    """
    # 1. Поиск релевантных чанков
    logger.info(f"Поиск по запросу: {question}")
    chunks = hybrid_search(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "Не удалось найти релевантную информацию в документах.",
            "sources": [],
            "chunks": [],
        }

    # 2. Формирование контекста
    context = format_context(chunks)

    # 3. Генерация ответа через LM Studio
    logger.info("Генерация ответа через LM Studio...")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question,
    })
    answer = response.content

    # 4. Сбор уникальных источников
    sources = []
    seen = set()
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": c["source"], "page": c["page"]})

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
    }


if __name__ == "__main__":
    test_questions = [
        "Какие параметры вибрации считаются критическими для газовой турбины?",
        "Что такое схема охлаждения лопаток?",
        "Какой ресурс до капитального ремонта у турбины V64.3A?",
    ]

    for q in test_questions:
        print(f"\n{'=' * 70}")  
        print(f"ВОПРОС: {q}")
        print('=' * 70)

        result = generate_answer(q, top_k=5)
        print(f"\nОТВЕТ:\n{result['answer']}")
        print(f"\nИСТОЧНИКИ:")
        for s in result["sources"]:
            print(f"  - {s['source']}, стр. {s['page']}")