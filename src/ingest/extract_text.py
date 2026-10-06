import pdfplumber
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Извлекает текст постранично из PDF.
    Возвращает список словарей: {page: int, text: str, source: str}
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        "page": i,
                        "text": text.strip(),
                        "source": pdf_path.name,
                        "total_pages": len(pdf.pages)
                    })
    except Exception as e:
        logger.error(f"Ошибка при обработке {pdf_path}: {e}")
        return []
    return pages

def process_all_pdfs(raw_dir: str, output_path: str):
    """Обрабатывает все PDF в папке и сохраняет результат в JSON."""
    raw_path = Path(raw_dir)
    all_pages = []

    pdf_files = list(raw_path.glob("*.pdf"))
    logger.info(f"Найдено {len(pdf_files)} PDF-файлов")

    for pdf_file in pdf_files:
        logger.info(f"Обработка: {pdf_file.name}")
        pages = extract_text_from_pdf(pdf_file)
        all_pages.extend(pages)

    # Сохраняем результат
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    logger.info(f"Сохранено {len(all_pages)} страниц в {output_path}")
    return all_pages

if __name__ == "__main__":
    process_all_pdfs(
        raw_dir="data/raw",
        output_path="data/processed/pages.json"
    )