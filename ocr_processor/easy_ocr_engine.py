import easyocr
from typing import List, Dict, Any
import logging
from pathlib import Path
from . import pdf_utils
from .base_engine import OCREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EasyOCREngine(OCREngine):
    def __init__(self, languages: List[str] = ["en"]):
        self.reader = easyocr.Reader(languages)

    def process_image(self, image_path: str) -> Dict[str, Any]:
        try:
            result = self.reader.readtext(image_path)
            text = " ".join([item[1] for item in result])
            confidence = sum([item[2] for item in result]) / len(result) if result else 0

            return {
                "engine": "easyocr",
                "text": text,
                "confidence": confidence,
                "boxes": [
                    {
                        "text": item[1],
                        "confidence": item[2],
                        "box": item[0]
                    }
                    for item in result
                ]
            }
        except Exception as e:
            logger.error(f"Error processing image with EasyOCR: {str(e)}")
            return {"error": str(e)}

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        try:
            # Convert PDF pages to images
            image_paths = pdf_utils.convert_pdf_to_images(pdf_path)
            
            # Process each image
            results = []
            for img_path in image_paths:
                result = self.process_image(str(img_path))
                results.append(result)
                
                # Clean up temporary image
                Path(img_path).unlink()
                
            return {
                "engine": "easyocr",
                "pages": results
            }
        except Exception as e:
            logger.error(f"Error processing PDF with EasyOCR: {str(e)}")
            return {"error": str(e)}
