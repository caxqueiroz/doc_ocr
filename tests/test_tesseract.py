from pathlib import Path
import pytest
from ocr_processor.ocr_engines import TesseractEngine


@pytest.fixture
def invoice_path() -> Path:
    """Fixture to provide the path to the test invoice image"""
    path = Path(__file__).parent.parent / "data" / "invoice.jpeg"
    assert path.exists(), f"Test invoice not found at {path}"
    return path


@pytest.fixture
def tesseract_engine() -> TesseractEngine:
    """Fixture to provide a configured TesseractEngine instance"""
    return TesseractEngine(lang="eng")


def test_tesseract_invoice_processing(tesseract_engine: TesseractEngine, invoice_path: Path) -> None:
    """Test that Tesseract can process an invoice image and extract text"""
    # Process the invoice image
    result = tesseract_engine.process_image(str(invoice_path))

    # Check basic structure of the result
    assert isinstance(result, dict)
    assert "engine" in result
    assert result["engine"] == "tesseract"
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


def test_tesseract_confidence_scores(tesseract_engine: TesseractEngine, invoice_path: Path) -> None:
    """Test that Tesseract returns confidence scores for extracted text"""
    result = tesseract_engine.process_image(str(invoice_path))

    # Check overall confidence score
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 100  # Tesseract confidence is 0-100%

    # Check word-level confidence scores
    assert "word_confidences" in result
    assert isinstance(result["word_confidences"], list)
    assert len(result["word_confidences"]) > 0
    
    # Check structure of word confidences
    for word, conf in result["word_confidences"]:
        assert isinstance(word, str)
        assert isinstance(conf, (int, float))
        assert conf == -1 or 0 <= conf <= 100  # -1 for no confidence, 0-100 for valid scores


def test_tesseract_error_handling(tesseract_engine: TesseractEngine) -> None:
    """Test that Tesseract properly handles errors with invalid input"""
    # Test with non-existent file
    result = tesseract_engine.process_image("nonexistent.jpg")
    assert "error" in result
    assert isinstance(result["error"], str)

    # Test with invalid file path
    result = tesseract_engine.process_image("")
    assert "error" in result
    assert isinstance(result["error"], str)
