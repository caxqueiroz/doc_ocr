from abc import ABC, abstractmethod
import easyocr
import pytesseract
from PIL import Image
import pdf2image
from typing import List, Dict, Any
import logging
import base64
import requests
from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCREngine(ABC):
    @abstractmethod
    def process_image(self, image_path: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        pass


class EasyOCREngine(OCREngine):
    def __init__(self, languages: List[str] = ["en"]):
        self.reader = easyocr.Reader(languages)

    def process_image(self, image_path: str) -> Dict[str, Any]:
        try:
            results = self.reader.readtext(image_path)
            return {
                "engine": "easyocr",
                "text": " ".join([result[1] for result in results]),
                "confidence": [result[2] for result in results],
                "boxes": [result[0] for result in results],
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

            return {
                "engine": "tesseract",
                "text": text,
                "confidence": data["conf"],
                "boxes": list(
                    zip(data["left"], data["top"], data["width"], data["height"])
                ),
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
        """
        Encode an image into a string that can be processed by the LLM
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image with the LLM and extract the text
        """
        try:
            image_base64 = self._encode_image(image_path)
            prompt = """Please look at this image and extract all the text content.
Structure the output as JSON with these guidelines:
                - Identify different sections or components
                - Use appropriate keys for different text elements
                - Maintain the hierarchical structure of the content
                - Include all visible text from the image"""

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": (f"data:image/jpeg;base64,{image_base64}"),
                            },
                        ],
                    }
                ],
                "stream": False,
            }
            response = requests.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()

            # Extract the response text from Ollama's response
            result = response.json()
            return {
                "engine": "ollama",
                "model": self.model_name,
                "text": result.get("message", {}).get("content", ""),
                "raw_response": result,
            }
        except Exception as e:
            logger.error(f"Error processing image with LLM: {str(e)}")
            return {"error": str(e)}

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        # TODO: Implement vision model processing for PDFs
        pass
