import logging
from pathlib import Path
from ocr_processor.ocr_engines import OllamaLLMEngine
from ocr_processor.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ollama_invoice():
    # Initialize the Ollama engine with the default model
    engine = OllamaLLMEngine(model_name=config.OLLAMA_DEFAULT_MODEL)
    
    # Get the absolute path to the invoice image
    image_path = Path(__file__).parent.parent / "data" / "invoice.jpeg"
    
    if not image_path.exists():
        logger.error(f"Test image not found at {image_path}")
        return
    
    logger.info(f"Processing image: {image_path}")
    
    try:
        # Test base64 encoding
        with open(image_path, 'rb') as f:
            import base64
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
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
    test_ollama_invoice()
