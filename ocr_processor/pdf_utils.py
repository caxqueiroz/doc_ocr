"""Utilities for PDF processing."""
from typing import Dict, Any, List, Optional
import io
from pathlib import Path
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LAParams
from PIL import Image
import pdf2image


def has_selectable_text(pdf_path: str) -> bool:
    """Check if a PDF has selectable text.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        True if the PDF has selectable text, False otherwise.
    """
    text = extract_text(pdf_path, laparams=LAParams())
    return bool(text.strip())


def extract_text_with_confidence(pdf_path: str) -> Dict[str, Any]:
    """Extract text and confidence scores from a PDF with selectable text.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Dict containing text and confidence scores for each page.
    """
    results = []
    
    for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=LAParams())):
        page_text = []
        page_confidences = []
        page_boxes = []
        
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                if text:
                    # Get bounding box coordinates
                    box = [
                        [element.x0, element.y0],  # top-left
                        [element.x1, element.y0],  # top-right
                        [element.x1, element.y1],  # bottom-right
                        [element.x0, element.y1],  # bottom-left
                    ]
                    
                    # Calculate average font size and assume high confidence for native text
                    confidence = 1.0  # Native text is assumed to be highly accurate
                    
                    page_text.append(text)
                    page_confidences.append(confidence)
                    page_boxes.append(box)
        
        results.append({
            "page": page_num + 1,
            "text": " ".join(page_text),
            "confidence": page_confidences,
            "boxes": page_boxes,
            "word_data": list(zip(page_text, page_confidences)),
        })
    
    return {"pages": results}


def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert a PDF to a list of PIL Images.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        List of PIL Images, one for each page.
    """
    return pdf2image.convert_from_path(pdf_path)
