from paddleocr import PaddleOCR
from typing import Dict, Any
import logging
from pathlib import Path
from . import pdf_utils
from .base_engine import OCREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaddleOCREngine(OCREngine):
    def __init__(self, languages: str = "en"):
        # Optimize for CPU performance
        self.reader = PaddleOCR(
            use_angle_cls=True,
            lang=languages,
            use_gpu=False,  # Use CPU
            cpu_threads=6,  # Utilize multiple CPU cores
            enable_mkldnn=False  # Disable MKL-DNN due to compatibility issues
        )

    def process_image(self, image_path: str) -> Dict[str, Any]:
        try:
            result = self.reader.ocr(image_path)
            
            # PaddleOCR returns a list of pages, but for single image we use first page
            boxes = result[0] if result else []
            
            # Extract text and confidence
            text_items = []
            confidences = []
            formatted_boxes = []
            
            for box in boxes:
                coords, (text, confidence) = box
                text_items.append(text)
                confidences.append(confidence)
                formatted_boxes.append({
                    "text": text,
                    "confidence": confidence,
                    "box": coords
                })
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "engine": "paddleocr",
                "text": " ".join(text_items),
                "confidence": avg_confidence,
                "boxes": formatted_boxes
            }
        except Exception as e:
            logger.error(f"Error processing image with PaddleOCR: {str(e)}")
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
                "engine": "paddleocr",
                "pages": results
            }
        except Exception as e:
            logger.error(f"Error processing PDF with PaddleOCR: {str(e)}")
            return {"error": str(e)}
