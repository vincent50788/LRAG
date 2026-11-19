"""Step 0, OCR: turn scans and images into the markdown that chunking reads.

OCR (Optical Character Recognition) reads the text in a picture. Tesseract does it on this computer.
OCR is not a RAG step. It only makes text. `rag.py ocr` writes data/<name>.md, and `rag.py index` does the rest.

OCR text has no structure, so the output has a title and one `## Page N` heading for each page.
The chunker needs the `##` headings. Thus each page becomes one chunk.
"""
import sys
from pathlib import Path

from .config import DATA_DIR, OCR_DPI, OCR_LANG, SCANS_DIR

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
MIN_TEXT = 20  # A PDF page with fewer characters than this has no text layer. It is a scan, so OCR reads it.


def read_pages(path):
    """Return the text of each page: [(text, how), ...]. "how" is "ocr" or "text layer"."""
    # These packages load here, not at the top. Then index, ask, and eval work without them.
    import pypdfium2
    import pytesseract
    from PIL import Image

    def ocr(image):
        return pytesseract.image_to_string(image, lang=OCR_LANG).strip()

    # checking suffix, open one page image and send to ocr
    if path.suffix.lower() in IMAGE_TYPES:
        return [(ocr(Image.open(path)), "ocr")]

    # not image but pdf, open pages and send to ocr
    pages = []
    for page in pypdfium2.PdfDocument(path):
        text = page.get_textpage().get_text_range().replace("\r\n", "\n").strip()
        if len(text) >= MIN_TEXT:  # The PDF has real text. Copy it. OCR would only add errors.
            pages.append((text, "text layer"))
        else:  # A scan. Draw the page as a picture (72 points = 1 inch), then OCR the picture.
            pages.append((ocr(page.render(scale=OCR_DPI / 72).to_pil()), "ocr"))
    return pages


def to_markdown(path, pages):
    """Make the markdown text: a title, then one `## Page N` section for each page that has text."""
    title = path.stem.replace("-", " ").replace("_", " ").title()
    note = f"<!-- Made by `rag.py ocr` from {path.name} (Tesseract, lang={OCR_LANG}, {OCR_DPI} DPI). Check the text for OCR mistakes. -->"
    body = "\n\n".join(f"## Page {n}\n{text}" for n, (text, _) in enumerate(pages, 1) if text)
    return f"# {title}\n\n{note}\n\n{body}\n"  # The chunker drops this note: it is before the first `##`.


def run(target=None):
    """OCR one file, or every PDF and image in a folder (default: scans/). Write each result to data/<name>.md."""
    target = Path(target) if target else SCANS_DIR
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(f for f in target.iterdir() if f.suffix.lower() in IMAGE_TYPES | {".pdf"})
    else:
        sys.exit(f"Not found: {target}")
    if not files:
        sys.exit(f"No PDF or image files in {target}")

    for f in files:
        try:
            pages = read_pages(f)
        except ImportError as e:
            sys.exit(f"Missing package ({e.name}). Run: .venv/bin/pip install pytesseract pypdfium2 pillow")
        except Exception as e:
            if type(e).__name__ == "TesseractNotFoundError":
                sys.exit("Tesseract is not installed. On macOS, run: brew install tesseract")
            raise
        out = DATA_DIR / f"{f.stem}.md"
        out.write_text(to_markdown(f, pages))
        by_ocr = sum(1 for _, how in pages if how == "ocr")
        chars = sum(len(text) for text, _ in pages)
        print(f"{f.name}: {len(pages)} pages ({by_ocr} read by OCR, {len(pages) - by_ocr} from the text layer), {chars} characters → {out.relative_to(out.parent.parent)}")
        for n, (text, _) in enumerate(pages, 1):
            if not text:
                print(f"  Warning: page {n} has no text. It makes no chunk.")
