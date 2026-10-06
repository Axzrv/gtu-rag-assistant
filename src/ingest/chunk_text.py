import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_pages(pages_path: str, output_path: str):
    """
    Разбивает страницы на чанки с сохранением метаданных.
    """
    with open(pages_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # Создаём Document-объекты (нужны для метаданных)
    documents = []
    for page in pages:
        doc = Document(
            page_content=page["text"],
            metadata={
                "source": page["source"],
                "page": page["page"],
                "total_pages": page.get("total_pages")
            }
        )
        documents.append(doc)

    # Сплиттер с настройками для технической документации
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,           # ~512 токенов для русского текста
        chunk_overlap=90,         # ~15% перекрытия
        length_function=len,
        separators=[
            "\n\n",               # абзацы
            "\n",                 # строки
            ". ",                 # предложения
            ", ",                 # части предложений
            " ",                  # слова
            ""                    # символы (крайний случай)
        ],
        is_separator_regex=False
    )

    chunks = splitter.split_documents(documents)

    # Добавляем ID и сохраняем
    output_chunks = []
    for i, chunk in enumerate(chunks):
        output_chunks.append({
            "id": f"chunk_{i:05d}",
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_chunks, f, ensure_ascii=False, indent=2)

    print(f"Создано {len(output_chunks)} чанков")
    print(f"Средний размер чанка: {sum(len(c['text']) for c in output_chunks) / len(output_chunks):.0f} символов")
    return output_chunks

if __name__ == "__main__":
    chunk_pages(
        pages_path="data/processed/pages.json",
        output_path="data/processed/chunks.json"
    )