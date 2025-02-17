import logging
from typing import Dict, Any
from pathlib import Path
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from . import pdf_utils
from .base_engine import OCREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuryaEngine(OCREngine):
    """OCR engine that uses Surya for text detection and recognition."""

    def __init__(self, model_name: str = "base", device: str = "cpu") -> None:
        """Initialize Surya OCR engine.

        Args:
            model_name: The name of the Surya model to use. Default is 'base'.
            device: Device to run inference on ('cpu' or 'cuda'). Default is 'cpu'.
        """
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()
        self.model_name = model_name
        self.device = device

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process an image with Surya OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict containing OCR results with text, confidence scores, and bounding boxes.
        """
        try:
            # Detect text regions
            detection_result = self.detection_predictor.predict(image_path)
            
            # Recognize text in detected regions
            recognition_result = self.recognition_predictor.predict(image_path)
            
            # Combine results
            text = " ".join([item["text"] for item in recognition_result])
            confidence = sum([item["confidence"] for item in recognition_result]) / len(recognition_result) if recognition_result else 0
            
            return {
                "engine": "surya",
                "model": self.model_name,
                "text": text,
                "confidence": confidence,
                "boxes": [
                    {
                        "text": item["text"],
                        "confidence": item["confidence"],
                        "box": item["box"]
                    }
                    for item in recognition_result
                ]
            }
            
        except Exception as e:
            logger.error(f"Error processing image with Surya: {str(e)}")
            return {"error": str(e)}

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF document with Surya OCR.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict containing OCR results for each page.
        """
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
                "engine": "surya",
                "model": self.model_name,
                "pages": results
            }
        except Exception as e:
            logger.error(f"Error processing PDF with Surya: {str(e)}")
            return {"error": str(e)}
