import base64
import requests
import logging
from typing import Dict, Any
from pathlib import Path
from . import pdf_utils
from .base_engine import OCREngine
from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaLLMEngine(OCREngine):
    def __init__(self) -> None:
        self.model_name = config.OLLAMA_DEFAULT_MODEL

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def process_image(self, image_path: str) -> Dict[str, Any]:
        try:
            image_base64 = self._encode_image(image_path)

            # Construct the prompt
            prompt = (
                "Extract text from this invoice image. Focus on key entities like:\n"
                "1. Order details (number, date)\n"
                "2. Items and products\n"
                "3. Dimensions and specifications\n"
                "4. Customer information\n"
                "5. Additional services\n"
                "6. Company information\n\n"
                "Return the extracted information in a clear, organized format."
            )

            # Make request to Ollama API
            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "images": [image_base64],
                },
            )

            if response.status_code != 200:
                raise Exception(f"Error from Ollama API: {response.text}")

            return {
                "engine": "ollama",
                "model": self.model_name,
                "text": response.json()["response"]
            }

        except Exception as e:
            logger.error(f"Error processing image with Ollama: {str(e)}")
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
                "engine": "ollama",
                "model": self.model_name,
                "pages": results
            }
        except Exception as e:
            logger.error(f"Error processing PDF with Ollama: {str(e)}")
            return {"error": str(e)}
