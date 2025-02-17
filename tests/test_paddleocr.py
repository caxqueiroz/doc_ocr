from pathlib import Path
import pytest
from ocr_processor.ocr_engines import PaddleOCREngine


@pytest.fixture
def invoice_path() -> Path:
    """Fixture to provide the path to the test invoice image"""
    path = Path(__file__).parent.parent / "data" / "invoice.jpeg"
    assert path.exists(), f"Test invoice not found at {path}"
    return path


@pytest.fixture
def paddleocr_engine() -> PaddleOCREngine:
    """Fixture to provide a configured PaddleOCREngine instance"""
    return PaddleOCREngine(languages="en")


def test_paddleocr_invoice_processing(
    paddleocr_engine: PaddleOCREngine, invoice_path: Path
) -> None:
    """Test that PaddleOCR can process an invoice image and extract text"""
    # Process the invoice image
    result = paddleocr_engine.process_image(str(invoice_path))

    # Check basic structure of the result
    assert isinstance(result, dict)
    assert "engine" in result
    assert result["engine"] == "paddleocr"
    assert "text" in result
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0
    
    # Get the extracted text
    text = result["text"]
    
    # 1. Order Information
    assert "85714" in text, "Order number not found"
    assert "february 2025" in text.lower(), "Order date not found"
    
    # 2. Product Information
    product_terms = [
        "scandinavian",
        "beech",
        "wood",
        "bunk",  # OCR sometimes reads 'bunk' instead of 'kids'
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
        ["w1200", "w12oo", "w12o0"],  # Account for OCR variations
        ["l1900", "l19oo", "l19o0"],
        ["super single"],
    ]
    for term_variants in dimension_terms:
        found = any(variant.lower() in text.lower() for variant in term_variants)
        assert found, f"No variant of {term_variants[0]} found in dimensions"
    
    # 4. Customer Information
    customer_terms = [
        "carlos queiroz",
        "kellock road",
        "singapore",
        "+6591937205",
    ]
    for term in customer_terms:
        assert term.lower() in text.lower(), f"Customer info '{term}' not found"
    
    # 5. Additional Services
    service_terms = [
        "carry-up",
        "disposal",
        "service",
        "level",
    ]
    for term in service_terms:
        assert term.lower() in text.lower(), f"Service term '{term}' not found"
    
    # 6. Company Information
    company_terms = [
        "loft home",
        "gambles crescent",
        "singapore",
    ]
    for term in company_terms:
        assert term.lower() in text.lower(), f"Company info '{term}' not found"


def test_paddleocr_confidence_and_boxes(
    paddleocr_engine: PaddleOCREngine, invoice_path: Path
) -> None:
    """Test that PaddleOCR provides confidence scores and bounding boxes"""
    # Process the invoice image
    result = paddleocr_engine.process_image(str(invoice_path))

    # Check confidence score
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1

    # Check bounding boxes
    assert "boxes" in result
    assert isinstance(result["boxes"], list)
    assert len(result["boxes"]) > 0

    # Check structure of each box
    for box in result["boxes"]:
        assert isinstance(box, dict)
        assert "text" in box
        assert isinstance(box["text"], str)
        assert "confidence" in box
        assert isinstance(box["confidence"], float)
        assert 0 <= box["confidence"] <= 1
        assert "box" in box
        assert isinstance(box["box"], list)
        assert len(box["box"]) == 4  # PaddleOCR uses 4 points for boxes


def test_paddleocr_error_handling(paddleocr_engine: PaddleOCREngine) -> None:
    """Test that PaddleOCR properly handles errors with invalid input"""
    # Test with non-existent file
    result = paddleocr_engine.process_image("nonexistent.jpg")
    assert "error" in result
    assert isinstance(result["error"], str)

    # Test with invalid file path
    result = paddleocr_engine.process_image("")
    assert "error" in result
    assert isinstance(result["error"], str)


def test_paddleocr_language_support(invoice_path: Path) -> None:
    """Test that PaddleOCR can be initialized with different languages"""
    # Test with English
    en_engine = PaddleOCREngine(languages="en")
    en_result = en_engine.process_image(str(invoice_path))
    assert isinstance(en_result, dict)
    assert "text" in en_result
    assert len(en_result["text"]) > 0

    # Test with Chinese
    ch_engine = PaddleOCREngine(languages="ch")
    ch_result = ch_engine.process_image(str(invoice_path))
    assert isinstance(ch_result, dict)
    assert "text" in ch_result
    assert len(ch_result["text"]) > 0
