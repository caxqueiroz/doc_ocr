import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from ner_processor.ner_engines import OpenAINEREngine, OllamaNEREngine, SpacyNEREngine
from ocr_processor.ocr_engines import GPT4VisionEngine

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

def test_ner_with_ocr():
    # Initialize OCR engine
    ocr_engine = GPT4VisionEngine()

    # Initialize NER engines
    ner_engines = [
        OpenAINEREngine(),
        OllamaNEREngine(),
        SpacyNEREngine()
    ]

    # Process image with OCR
    image_path = str(Path(__file__).parent.parent / 'data' / 'invoice.jpeg')
    logger.info(f"Processing image: {image_path}")
    logger.info("-" * 50)

    ocr_result = ocr_engine.process_image(image_path)
    if "error" in ocr_result:
        logger.error(f"OCR Error: {ocr_result['error']}")
        return

    extracted_text = ocr_result.get("text", "")
    logger.info("OCR Result:")
    logger.info("-" * 50)
    logger.info(extracted_text)
    logger.info("-" * 50)

    # Process the extracted text with each NER engine
    results = []
    for engine in ner_engines:
        logger.info(f"\nProcessing with {engine.__class__.__name__}:")
        logger.info("-" * 50)
        result = engine.process_text(extracted_text)
        results.append(result)

        # Log the result
        if "error" in result:
            logger.error(f"Error: {result['error']}")
        else:
            logger.info(f"Entities extracted: {json.dumps(result['entities'], indent=2)}")
            logger.info(f"Model used: {result['model']}")
        logger.info("-" * 50)

    return results

if __name__ == "__main__":
    test_ner_with_ocr()
