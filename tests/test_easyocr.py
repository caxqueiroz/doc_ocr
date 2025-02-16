from pathlib import Path
import pytest
from ocr_processor.ocr_engines import EasyOCREngine


@pytest.fixture
def invoice_path() -> Path:
    """Fixture to provide the path to the test invoice image"""
    path = Path(__file__).parent.parent / "data" / "invoice.jpeg"
    assert path.exists(), f"Test invoice not found at {path}"
    return path


@pytest.fixture
def easyocr_engine() -> EasyOCREngine:
    """Fixture to provide a configured EasyOCREngine instance"""
    return EasyOCREngine(languages=["en"])


def test_easyocr_invoice_processing(
    easyocr_engine: EasyOCREngine, invoice_path: Path
) -> None:
    """Test that EasyOCR can process an invoice image and extract text"""
    # Process the invoice image
    result = easyocr_engine.process_image(str(invoice_path))

    # Check basic structure of the result
    assert isinstance(result, dict)
    assert "engine" in result
    assert result["engine"] == "easyocr"
    assert "text" in result
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0

    # Check for expected invoice content
    text = result["text"].lower()
    expected_terms = [
        "order",
        "list",
        "ship",
        "quantity",
        "mattress",
        "bed",
        "singapore",
    ]
    for term in expected_terms:
        assert term in text, f"Expected to find '{term}' in extracted text"


def test_easyocr_confidence_and_boxes(
    easyocr_engine: EasyOCREngine, invoice_path: Path
) -> None:
    """Test that EasyOCR returns confidence scores and bounding boxes"""
    result = easyocr_engine.process_image(str(invoice_path))

    # Check confidence scores
    assert "confidence" in result
    assert isinstance(result["confidence"], list)
    assert len(result["confidence"]) > 0

    # Verify confidence scores are between 0 and 1 (EasyOCR uses 0-1 range)
    for conf in result["confidence"]:
        assert isinstance(conf, float)
        assert 0 <= conf <= 1, f"Confidence score {conf} not in range [0,1]"

    # Check bounding boxes
    assert "boxes" in result
    assert isinstance(result["boxes"], list)
    assert len(result["boxes"]) > 0

    # Verify box structure (each box should be a list of 4 points)
    for box in result["boxes"]:
        assert isinstance(box, list)
        assert len(box) == 4, "Each box should have 4 corner points"
        for point in box:
            assert isinstance(point, list)
            assert len(point) == 2, "Each point should have x,y coordinates"
            assert all(isinstance(coord, (int, float)) for coord in point)


def test_easyocr_error_handling(easyocr_engine: EasyOCREngine) -> None:
    """Test that EasyOCR properly handles errors with invalid input"""
    # Test with non-existent file
    result = easyocr_engine.process_image("nonexistent.jpg")
    assert "error" in result
    assert isinstance(result["error"], str)

    # Test with invalid file path
    result = easyocr_engine.process_image("")
    assert "error" in result
    assert isinstance(result["error"], str)


def test_easyocr_language_support(invoice_path: Path) -> None:
    """Test that EasyOCR can be initialized with different languages"""
    # Test with multiple languages
    multi_lang_engine = EasyOCREngine(languages=["en", "ch_sim"])
    result = multi_lang_engine.process_image(str(invoice_path))
    assert isinstance(result, dict)
    assert "error" not in result

    # Test with single language
    single_lang_engine = EasyOCREngine(languages=["en"])
    result = single_lang_engine.process_image(str(invoice_path))
    assert isinstance(result, dict)
    assert "error" not in result
