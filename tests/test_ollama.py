import logging
from pathlib import Path
from ocr_processor.ocr_engines import OllamaLLMEngine


# Configure logging to show all levels and output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def test_ollama_invoice() -> None:
    # Initialize the Ollama engine with default model from config
    engine = OllamaLLMEngine()

    # Get the absolute path to the invoice image
    image_path = Path(__file__).parent.parent / "data" / "invoice.jpeg"

    if not image_path.exists():
        logger.error(f"Test image not found at {image_path}")
        return

    logger.info(f"Processing image: {image_path}")

    # Process the image
    result = engine.process_image(str(image_path))

    # Verify result structure
    assert "error" not in result, f"Error in OCR: {result.get('error')}"
    assert "engine" in result, "Missing 'engine' field in result"
    assert result["engine"] == "ollama", f"Wrong engine: {result['engine']}"
    assert "model" in result, "Missing 'model' field in result"
    assert "text" in result, "Missing 'text' field in result"

    # Log the raw output for debugging
    logger.info("Raw model output:")
    logger.info(result["text"])

    # Get the extracted text
    text = result["text"]
    
    # Log the OCR result
    logger.info("OCR Result:")
    logger.info("-" * 50)
    logger.info(f"Engine: {result['engine']}")
    logger.info(f"Model: {result['model']}")
    logger.info("Extracted Information:")
    
    # 1. Order Information
    assert "85714" in text, "Order number not found"
    assert "february 2025" in text.lower(), "Order date not found"
    
    # 2. Product Information
    product_terms = [
        "scandinavian",
        "beech",
        "wood",
        "bunk",  # Sometimes reads 'bunk' instead of 'kids'
        "bed",
        "trundle",
        "cubble",
        "mattress",
        "hybrid",
        "medium",
        "firm",
        "zest",
    ]
    found_terms = sum(1 for term in product_terms if term.lower() in text.lower())
    assert found_terms >= len(product_terms) * 0.8, f"Only found {found_terms} out of {len(product_terms)} product terms"
    
    # 3. Dimensions and Specifications
    dimension_terms = [
        "w1200",
        "l1900",
        "super single",
    ]
    found_dims = sum(1 for term in dimension_terms if term.lower() in text.lower())
    assert found_dims >= 2, f"Only found {found_dims} out of {len(dimension_terms)} dimension terms"
    
    # 4. Customer Information
    customer_terms = [
        "carlos queiroz",
        "kellock road",
        "singapore",
        "+6591937205",
    ]
    found_customer = sum(1 for term in customer_terms if term.lower() in text.lower())
    assert found_customer >= len(customer_terms) * 0.75, f"Only found {found_customer} out of {len(customer_terms)} customer terms"
    
    # 5. Additional Services
    service_terms = [
        "carry",
        "disposal",
        "service",
        "level",
    ]
    found_services = sum(1 for term in service_terms if term.lower() in text.lower())
    assert found_services >= len(service_terms) * 0.75, f"Only found {found_services} out of {len(service_terms)} service terms"
    
    # 6. Company Information
    company_terms = [
        "loft home",
        "crescent",  # Could be Gambles or Gambas
        "singapore",
    ]
    found_company = sum(1 for term in company_terms if term.lower() in text.lower())
    assert found_company >= len(company_terms) * 0.75, f"Only found {found_company} out of {len(company_terms)} company terms"
    
    logger.info("-" * 50)
    logger.error(f"Error processing image: {str(e)}")
    

if __name__ == "__main__":
    test_ollama_invoice()
