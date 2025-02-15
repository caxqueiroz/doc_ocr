import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from ocr_processor.ocr_engines import GPT4VisionEngine
from ocr_processor.config import config

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
with open(env_path) as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY='):
            key = line.split('=', 1)[1].strip()
            os.environ['OPENAI_API_KEY'] = key
            break

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gpt4_vision_invoice():
    # Initialize the GPT-4 Vision engine with model from config
    engine = GPT4VisionEngine()

    # Get the absolute path to the invoice image
    image_path = Path(__file__).parent.parent / "data" / "invoice.jpeg"

    if not image_path.exists():
        logger.error(f"Test image not found at {image_path}")
        return

    logger.info(f"Processing image: {image_path}")

    try:
        # Test base64 encoding
        with open(image_path, "rb") as f:
            import base64
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
            logger.info(f"Base64 length: {len(image_base64)}")
            logger.info(f"First 100 chars: {image_base64[:100]}")

        # Process the image
        result = engine.process_image(str(image_path))

        # Log the result
        logger.info("OCR Result:")
        logger.info("-" * 50)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
        else:
            logger.info(f"Text extracted: {result['text']}")
            logger.info(f"Model used: {result['model']}")
        logger.info("-" * 50)

        return result
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


if __name__ == "__main__":
    test_gpt4_vision_invoice()
