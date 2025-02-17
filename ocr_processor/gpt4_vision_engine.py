import base64
import logging
from typing import Dict, Any
from pathlib import Path
from openai import OpenAI
from . import pdf_utils
from .base_engine import OCREngine
from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT4VisionEngine(OCREngine):
    """OCR engine that uses GPT-4 Vision API"""

    def __init__(self) -> None:
        # Clean up API key
        api_key = config.OPENAI_API_KEY.strip() if config.OPENAI_API_KEY else None
        logger.info(f"API key present: {bool(api_key)}")

        self.client = OpenAI(api_key=api_key)
        self.model = config.OPENAI_MODEL

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def process_image(self, image_path: str) -> Dict[str, Any]:
        try:
            image_base64 = self._encode_image(image_path)

            # Construct the system prompt
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

            # Make request to OpenAI API
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
                                "text": prompt
                            },
                        ],
                    }
                ],
            )

            return {
                "engine": "gpt4-vision",
                "model": self.model,
                "text": response.choices[0].message.content
            }

        except Exception as e:
            logger.error(f"Error processing image with GPT-4 Vision: {str(e)}")
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
                "engine": "gpt4-vision",
                "model": self.model,
                "pages": results
            }
        except Exception as e:
            logger.error(f"Error processing PDF with GPT-4 Vision: {str(e)}")
            return {"error": str(e)}
