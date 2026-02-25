"""
OCR utilities for extracting text from PDF files.
Supports both native text extraction and OCR fallback for scanned documents.
"""

import logging
from io import BytesIO

logger = logging.getLogger(__name__)


def extract_text_native(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF2 (native text extraction)."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes))
        pages_text = []

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                pages_text.append(text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i}: {e}")
                continue

        return "\n\n".join(pages_text)

    except Exception as e:
        logger.error(f"Native PDF extraction failed: {e}")
        return ""


def extract_text_ocr(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF using OCR (for scanned documents).
    Requires: tesseract, poppler-utils (system packages)
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError as e:
        logger.warning(f"OCR dependencies not available: {e}")
        return ""

    try:
        # Convert PDF pages to images
        images = convert_from_bytes(
            pdf_bytes,
            dpi=200,  # Balance between quality and speed
            fmt="jpeg",
        )

        pages_text = []
        for i, image in enumerate(images):
            try:
                # Run OCR on each page image
                text = pytesseract.image_to_string(
                    image,
                    lang="eng",
                    config="--psm 1",  # Automatic page segmentation with OSD
                )
                pages_text.append(text)
            except Exception as e:
                logger.warning(f"OCR failed for page {i}: {e}")
                continue
            finally:
                # Free memory
                image.close()

        return "\n\n".join(pages_text)

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return ""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF file.

    Strategy:
    1. Try native text extraction first (fast, works for digital PDFs)
    2. If that yields little text, fall back to OCR (slower, for scanned docs)
    """
    # Try native extraction first
    native_text = extract_text_native(pdf_bytes)
    native_text_clean = native_text.strip()

    # If we got enough text, use it
    if len(native_text_clean) > 100:
        logger.info(f"Native extraction succeeded: {len(native_text_clean)} chars")
        return native_text

    # Otherwise try OCR
    logger.info("Native extraction insufficient, trying OCR...")
    ocr_text = extract_text_ocr(pdf_bytes)
    ocr_text_clean = ocr_text.strip()

    # Return whichever has more content
    if len(ocr_text_clean) > len(native_text_clean):
        logger.info(f"OCR extraction used: {len(ocr_text_clean)} chars")
        return ocr_text

    logger.info(f"Using native text: {len(native_text_clean)} chars")
    return native_text
