from abc import ABC, abstractmethod
import easyocr
import pytesseract
import numpy as np
from PIL import Image
import pdf2image
from typing import List, Dict, Any
import logging
import base64
import requests
from openai import OpenAI
from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCREngine(ABC):
    @abstractmethod
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process an image and extract text

        Args:
            image_path (str): Path to the image file

        Returns:
            dict: JSON schema with extracted text
        """
        raise NotImplementedError

    @abstractmethod
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF and extract text

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            dict: JSON schema with extracted text
        """
        raise NotImplementedError


class EasyOCREngine(OCREngine):
    def __init__(self, languages: List[str] = ["en"]):
        self.reader = easyocr.Reader(languages)

    def process_image(self, image_path: str) -> Dict[str, Any]:
        try:
            results = self.reader.readtext(image_path)
            # Convert bounding boxes to a consistent format
            boxes = []
            for result in results:
                box = result[0]  # Each box is a list of 4 points
                converted_box = []
                for point in box:
                    # Each point is [x, y]
                    x, y = point
                    converted_box.append([
                        float(x) if isinstance(x, (int, float, np.integer, np.floating)) else 0.0,
                        float(y) if isinstance(y, (int, float, np.integer, np.floating)) else 0.0
                    ])
                boxes.append(converted_box)

            return {
                "engine": "easyocr",
                "text": " ".join([result[1] for result in results]),
                "confidence": [float(result[2]) for result in results],
                "boxes": boxes,
            }
        except Exception as e:
            logger.error(f"Error processing image with EasyOCR: {str(e)}")
            return {"error": str(e)}

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        try:
            pages = pdf2image.convert_from_path(pdf_path)
            results = []
            for i, page in enumerate(pages):
                page_result = self.reader.readtext(page)
                results.append(
                    {
                        "page": i + 1,
                        "text": " ".join([result[1] for result in page_result]),
                        "confidence": [result[2] for result in page_result],
                        "boxes": [result[0] for result in page_result],
                    }
                )
            return {"engine": "easyocr", "pages": results}
        except Exception as e:
            logger.error(f"Error processing PDF with EasyOCR: {str(e)}")
            return {"error": str(e)}


class TesseractEngine(OCREngine):
    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def process_image(self, image_path: str) -> Dict[str, Any]:
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=self.lang)
            data = pytesseract.image_to_data(
                image, lang=self.lang, output_type=pytesseract.Output.DICT
            )

            # Calculate average confidence, excluding -1 values (which indicate no confidence)
            confidences = [conf for conf in data["conf"] if conf != -1]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return {
                "engine": "tesseract",
                "text": text,
                "confidence": avg_confidence,
                "boxes": list(
                    zip(data["left"], data["top"], data["width"], data["height"])
                ),
                "word_confidences": list(zip(data["text"], data["conf"])),
            }
        except Exception as e:
            logger.error(f"Error processing image with Tesseract: {str(e)}")
            return {"error": str(e)}

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        try:
            pages = pdf2image.convert_from_path(pdf_path)
            results = []
            for i, page in enumerate(pages):
                text = pytesseract.image_to_string(page, lang=self.lang)
                data = pytesseract.image_to_data(
                    page, lang=self.lang, output_type=pytesseract.Output.DICT
                )
                results.append(
                    {
                        "page": i + 1,
                        "text": text,
                        "confidence": data["conf"],
                        "boxes": list(
                            zip(
                                data["left"], data["top"], data["width"], data["height"]
                            )
                        ),
                    }
                )
            return {"engine": "tesseract", "pages": results}
        except Exception as e:
            logger.error(f"Error processing PDF with Tesseract: {str(e)}")
            return {"error": str(e)}


# Placeholder for future Vision LLM implementations
class OllamaLLMEngine(OCREngine):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image with the LLM and extract the text
        """
        try:
            image_base64 = self._encode_image(image_path)
            prompt = (
                "Extract all text from this image. "
                "Do not interpret the text, just extract it. "
                "Use JSON format"
            )

            # Create request with the image and prompt
            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "images": [image_base64],
                },
            )
            response.raise_for_status()
            result = response.json()

            return {
                "engine": "ollama",
                "model": self.model_name,
                "text": result.get("response", ""),
                "raw_response": result,
            }
        except Exception as e:
            logger.error(f"Error processing image with LLM: {str(e)}")
            return {"error": str(e)}

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        # TODO: Implement vision model processing for PDFs
        raise NotImplementedError("PDF processing not yet implemented for this engine")


class GPT4VisionEngine(OCREngine):
    """OCR engine that uses GPT-4 Vision API"""

    def __init__(self) -> None:
        # Clean up API key
        api_key = config.OPENAI_API_KEY.strip() if config.OPENAI_API_KEY else None
        logger.info(f"API key present: {bool(api_key)}")

        self.client = OpenAI(api_key=api_key)
        self.model = config.OPENAI_MODEL

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process an image with GPT-4 Vision API"""
        try:
            image_base64 = self._encode_image(image_path)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Extract all text from this image. "
                                    "Do not interpret the text, just extract it. "
                                    "Use JSON format"
                                ),
                            },
                        ],
                    }
                ],
                temperature=0.5,
                max_tokens=1000,
            )

            return {
                "engine": "gpt4-vision",
                "model": self.model,
                "text": response.choices[0].message.content,
                "raw_response": response.model_dump(),
            }

        except Exception as e:
            logger.error(f"Error processing image with GPT-4 Vision: {str(e)}")
            return {"error": str(e)}

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        # TODO: Implement PDF processing
        raise NotImplementedError("PDF processing not yet implemented for GPT4Vision")
