from pathlib import Path
import pytest
from PIL import Image
from ocr_processor.ocr_engines import SuryaEngine


@pytest.fixture
def invoice_path() -> Path:
    """Fixture to provide the path to the test invoice image"""
    path = Path(__file__).parent.parent / "data" / "invoice.jpeg"
    assert path.exists(), f"Test invoice not found at {path}"
    return path


@pytest.fixture
def surya_engine() -> SuryaEngine:
    """Fixture to provide a configured SuryaEngine instance"""
    return SuryaEngine(model_name="base", device="cpu")


def test_surya_invoice_processing(
    surya_engine: SuryaEngine, invoice_path: Path
) -> None:
    """Test that Surya can process an invoice image and extract text"""
    # Process the invoice image
    result = surya_engine.process_image(str(invoice_path))

    # Check basic structure of the result
    assert isinstance(result, dict)
    assert "engine" in result
    assert result["engine"] == "surya"
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


def test_surya_confidence_and_boxes(
    surya_engine: SuryaEngine, invoice_path: Path
) -> None:
    """Test that Surya returns confidence scores and bounding boxes"""
    result = surya_engine.process_image(str(invoice_path))

    # Check confidence scores
    assert "confidence" in result
    assert isinstance(result["confidence"], list)
    assert len(result["confidence"]) > 0

    # Verify confidence scores are between 0 and 1
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
            assert all(isinstance(coord, float) for coord in point)

    # Check word-level data
    assert "word_data" in result
    assert isinstance(result["word_data"], list)
    assert len(result["word_data"]) > 0
    for word, conf in result["word_data"]:
        assert isinstance(word, str)
        assert isinstance(conf, float)
        assert 0 <= conf <= 1


def test_surya_error_handling(surya_engine: SuryaEngine) -> None:
    """Test that Surya properly handles errors with invalid input"""
    # Test with non-existent file
    result = surya_engine.process_image("nonexistent.jpg")
    assert "error" in result
    assert isinstance(result["error"], str)

    # Test with invalid file path
    result = surya_engine.process_image("")
    assert "error" in result
    assert isinstance(result["error"], str)


def test_surya_pdf_processing(surya_engine: SuryaEngine, tmp_path: Path) -> None:
    """Test that Surya can process PDF files"""
    # Create a simple PDF for testing
    pdf_path = tmp_path / "test.pdf"
    test_image_path = Path(__file__).parent.parent / "data" / "invoice.jpeg"

    # Convert test image to PDF
    image = Image.open(test_image_path)
    image.save(pdf_path, "PDF")

    # Process the PDF
    result = surya_engine.process_pdf(str(pdf_path))

    # Check basic structure
    assert isinstance(result, dict)
    assert "engine" in result
    assert result["engine"] == "surya"
    assert "pages" in result
    assert isinstance(result["pages"], list)
    assert len(result["pages"]) > 0

    # Check first page
    page = result["pages"][0]
    assert isinstance(page, dict)
    assert "page" in page
    assert page["page"] == 1
    assert "text" in page
    assert isinstance(page["text"], str)
    assert len(page["text"]) > 0
    assert "confidence" in page
    assert isinstance(page["confidence"], list)
    assert "boxes" in page
    assert isinstance(page["boxes"], list)
    assert "word_data" in page
    assert isinstance(page["word_data"], list)
