from abc import ABC, abstractmethod
import easyocr
import pytesseract
import numpy as np
from PIL import Image
from typing import List, Dict, Any
import logging
import base64
import requests
from openai import OpenAI
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from . import pdf_utils
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
                    converted_box.append(
                        [
                            (
                                float(x)
                                if isinstance(x, (int, float, np.integer, np.floating))
                                else 0.0
                            ),
                            (
                                float(y)
                                if isinstance(y, (int, float, np.integer, np.floating))
                                else 0.0
                            ),
                        ]
                    )
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
            # First check if PDF has selectable text
            if pdf_utils.has_selectable_text(pdf_path):
                results = pdf_utils.extract_text_with_confidence(pdf_path)
                return {"engine": "native_pdf", **results}

            # If no selectable text, process with OCR
            pages = pdf_utils.convert_pdf_to_images(pdf_path)
            page_results: List[Dict[str, Any]] = []
            for i, page in enumerate(pages):
                # Save page to temporary file
                temp_path = f"/tmp/page_{i}.png"
                page.save(temp_path)
                page_result = self.reader.readtext(temp_path)
                page_results.append(
                    {
                        "page": i + 1,
                        "text": " ".join([result[1] for result in page_result]),
                        "confidence": [result[2] for result in page_result],
                        "boxes": [result[0] for result in page_result],
                    }
                )
            return {"engine": "easyocr", "pages": page_results}
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
            # First check if PDF has selectable text
            if pdf_utils.has_selectable_text(pdf_path):
                results = pdf_utils.extract_text_with_confidence(pdf_path)
                return {"engine": "native_pdf", **results}

            # If no selectable text, process with OCR
            pages = pdf_utils.convert_pdf_to_images(pdf_path)
            page_results: List[Dict[str, Any]] = []
            for i, page in enumerate(pages):
                text = pytesseract.image_to_string(page, lang=self.lang)
                data = pytesseract.image_to_data(
                    page, lang=self.lang, output_type=pytesseract.Output.DICT
                )
                page_results.append(
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
            return {"engine": "tesseract", "pages": page_results}
        except Exception as e:
            logger.error(f"Error processing PDF with Tesseract: {str(e)}")
            return {"error": str(e)}


# Placeholder for future Vision LLM implementations
class OllamaLLMEngine(OCREngine):
    def __init__(self) -> None:
        self.model_name = config.OLLAMA_DEFAULT_MODEL

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
        try:
            # First check if PDF has selectable text
            if pdf_utils.has_selectable_text(pdf_path):
                results = pdf_utils.extract_text_with_confidence(pdf_path)
                return {"engine": "native_pdf", **results}

            # If no selectable text, process with OCR
            pages = pdf_utils.convert_pdf_to_images(pdf_path)
            page_results: List[Dict[str, Any]] = []

            for i, page in enumerate(pages):
                # Save page to temporary file
                temp_path = f"/tmp/page_{i}.png"
                page.save(temp_path)
                # Process this page with the LLM
                result = self.process_image(temp_path)
                if "error" in result:
                    raise Exception(result["error"])

                page_results.append(
                    {
                        "page": i + 1,
                        "text": result["text"],
                        "confidence": 1.0,  # LLM doesn't provide confidence scores
                        "boxes": [],  # LLM doesn't provide bounding boxes
                        "raw_response": result.get("raw_response"),
                    }
                )

            return {"engine": "ollama", "model": self.model_name, "pages": page_results}

        except Exception as e:
            logger.error(f"Error processing PDF with LLM: {str(e)}")
            return {"error": str(e)}


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

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process an image with Surya OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict containing OCR results with text, confidence scores, and bounding boxes.
        """
        try:
            # Load and process the image
            image = Image.open(image_path)
            predictions = self.recognition_predictor(
                [image], [None], self.detection_predictor
            )
            result = predictions[0]  # Get first image's results

            # Extract text, confidence scores, and bounding boxes
            texts = []
            confidences = []
            boxes = []

            for text_line in result.text_lines:
                texts.append(text_line.text)
                confidences.append(float(text_line.confidence))
                # Convert box coordinates to our standard format
                box = [
                    [float(x), float(y)]
                    for x, y in [
                        (text_line.bbox[0], text_line.bbox[1]),  # top-left
                        (text_line.bbox[2], text_line.bbox[1]),  # top-right
                        (text_line.bbox[2], text_line.bbox[3]),  # bottom-right
                        (text_line.bbox[0], text_line.bbox[3]),  # bottom-left
                    ]
                ]
                boxes.append(box)

            return {
                "engine": "surya",
                "text": " ".join(texts),
                "confidence": confidences,
                "boxes": boxes,
                "word_data": list(zip(texts, confidences)),
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
            # First check if PDF has selectable text
            if pdf_utils.has_selectable_text(pdf_path):
                results = pdf_utils.extract_text_with_confidence(pdf_path)
                return {"engine": "native_pdf", **results}

            # If no selectable text, process with OCR
            pages = pdf_utils.convert_pdf_to_images(pdf_path)
            predictions = self.recognition_predictor(
                pages, [None] * len(pages), self.detection_predictor
            )
            page_results: List[Dict[str, Any]] = []

            for i, result in enumerate(predictions):
                texts: List[str] = []
                confidences: List[float] = []
                boxes: List[List[List[float]]] = []

                for text_line in result.text_lines:
                    texts.append(text_line.text)
                    confidences.append(float(text_line.confidence))
                    box = [
                        [float(x), float(y)]
                        for x, y in [
                            (text_line.bbox[0], text_line.bbox[1]),
                            (text_line.bbox[2], text_line.bbox[1]),
                            (text_line.bbox[2], text_line.bbox[3]),
                            (text_line.bbox[0], text_line.bbox[3]),
                        ]
                    ]
                    boxes.append(box)

                page_results.append(
                    {
                        "page": i + 1,
                        "text": " ".join(texts),
                        "confidence": confidences,
                        "boxes": boxes,
                        "word_data": list(zip(texts, confidences)),
                    }
                )

            return {"engine": "surya", "pages": page_results}

        except Exception as e:
            logger.error(f"Error processing PDF with Surya: {str(e)}")
            return {"error": str(e)}


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
        try:
            # First check if PDF has selectable text
            if pdf_utils.has_selectable_text(pdf_path):
                results = pdf_utils.extract_text_with_confidence(pdf_path)
                return {"engine": "native_pdf", **results}

            # If no selectable text, process with OCR
            pages = pdf_utils.convert_pdf_to_images(pdf_path)
            page_results: List[Dict[str, Any]] = []

            for i, page in enumerate(pages):
                # Save page to temporary file
                temp_path = f"/tmp/page_{i}.png"
                page.save(temp_path)
                # Process with GPT-4 Vision
                image_base64 = self._encode_image(temp_path)

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

                page_results.append(
                    {
                        "page": i + 1,
                        "text": response.choices[0].message.content,
                        "confidence": 1.0,  # GPT-4 Vision doesn't provide confidence scores
                        "boxes": [],  # GPT-4 Vision doesn't provide bounding boxes
                        "raw_response": response.model_dump(),
                    }
                )

            return {"engine": "gpt4-vision", "model": self.model, "pages": page_results}

        except Exception as e:
            logger.error(f"Error processing PDF with GPT-4 Vision: {str(e)}")
            return {"error": str(e)}
