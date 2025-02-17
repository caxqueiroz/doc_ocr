import pytesseract
import logging
from typing import Dict, Any
from pathlib import Path
from PIL import Image
from . import pdf_utils
from .base_engine import OCREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TesseractEngine(OCREngine):
    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def process_image(self, image_path: str) -> Dict[str, Any]:
        try:
            # Open image with PIL
            image = Image.open(image_path)
            
            # Get OCR data
            data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)
            
            # Extract text and confidence scores
            words = []
            confidences = []
            boxes = []
            
            for i in range(len(data["text"])):
                if data["text"][i].strip():
                    words.append(data["text"][i])
                    confidences.append(data["conf"][i])
                    boxes.append([
                        data["left"][i],
                        data["top"][i],
                        data["left"][i] + data["width"][i],
                        data["top"][i] + data["height"][i]
                    ])
            
            text = " ".join(words)
            confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "engine": "tesseract",
                "text": text,
                "confidence": confidence,
                "boxes": [
                    {
                        "text": word,
                        "confidence": conf,
                        "box": box
                    }
                    for word, conf, box in zip(words, confidences, boxes)
                ]
            }
            
        except Exception as e:
            logger.error(f"Error processing image with Tesseract: {str(e)}")
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
                "engine": "tesseract",
                "pages": results
            }
        except Exception as e:
            logger.error(f"Error processing PDF with Tesseract: {str(e)}")
            return {"error": str(e)}
