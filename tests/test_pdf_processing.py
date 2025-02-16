"""Test PDF processing capabilities of all OCR engines."""
import os
from pathlib import Path
import pytest
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from ocr_processor.ocr_engines import (
    EasyOCREngine,
    TesseractEngine,
    OllamaLLMEngine,
    GPT4VisionEngine,
    SuryaEngine,
)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Fixture to provide the path to the test data directory"""
    path = Path(__file__).parent.parent / "data"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture(scope="session")
def invoice_path() -> Path:
    """Fixture to provide the path to the test invoice image"""
    path = Path(__file__).parent.parent / "data" / "invoice.jpeg"
    assert path.exists(), f"Test invoice not found at {path}"
    return path


@pytest.fixture(scope="session")
def selectable_pdf(test_data_dir: Path) -> Path:
    """Create a PDF with selectable text for testing."""
    pdf_path = test_data_dir / "selectable_test.pdf"
    
    # Create PDF with selectable text
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "This is a test PDF with selectable text")
    c.drawString(100, 700, "Page 1 - Sample Invoice")
    c.drawString(100, 650, "Item 1: $100")
    c.drawString(100, 600, "Item 2: $200")
    c.drawString(100, 550, "Total: $300")
    c.showPage()
    
    # Add a second page
    c.drawString(100, 750, "Page 2 - Additional Information")
    c.drawString(100, 700, "Customer: John Doe")
    c.drawString(100, 650, "Order ID: 12345")
    c.drawString(100, 600, "Shipping Address: 123 Test St")
    c.showPage()
    c.save()
    
    return pdf_path


@pytest.fixture(scope="session")
def image_only_pdf(test_data_dir: Path, invoice_path: Path) -> Path:
    """Create a PDF with only image content (no selectable text) for testing."""
    pdf_path = test_data_dir / "image_only_test.pdf"
    
    # Convert the invoice image to PDF
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawImage(str(invoice_path), 0, 0, width=600, height=800)
    c.showPage()
    c.save()
    
    return pdf_path


@pytest.fixture
def easyocr_engine() -> EasyOCREngine:
    """Fixture to provide a configured EasyOCREngine instance"""
    return EasyOCREngine(languages=["en"])


@pytest.fixture
def tesseract_engine() -> TesseractEngine:
    """Fixture to provide a configured TesseractEngine instance"""
    return TesseractEngine(lang="eng")


@pytest.fixture
def ollama_engine() -> OllamaLLMEngine:
    """Fixture to provide a configured OllamaLLMEngine instance"""
    return OllamaLLMEngine()


@pytest.fixture
def gpt4_engine() -> GPT4VisionEngine:
    """Fixture to provide a configured GPT4VisionEngine instance"""
    return GPT4VisionEngine()


@pytest.fixture
def surya_engine() -> SuryaEngine:
    """Fixture to provide a configured SuryaEngine instance"""
    return SuryaEngine()


def test_easyocr_selectable_pdf(easyocr_engine: EasyOCREngine, selectable_pdf: Path) -> None:
    """Test that EasyOCR can process a PDF with selectable text."""
    result = easyocr_engine.process_pdf(str(selectable_pdf))
    
    # Check basic structure
    assert isinstance(result, dict)
    assert "engine" in result
    assert result["engine"] == "native_pdf"  # Should use native PDF extraction
    assert "pages" in result
    assert isinstance(result["pages"], list)
    assert len(result["pages"]) == 2  # Should have 2 pages
    
    # Check content of first page
    page1 = result["pages"][0]
    assert isinstance(page1, dict)
    assert "text" in page1
    text = page1["text"].lower()
    assert "test pdf" in text
    assert "sample invoice" in text
    assert "item 1: $100" in text
    
    # Check content of second page
    page2 = result["pages"][1]
    text = page2["text"].lower()
    assert "additional information" in text
    assert "john doe" in text
    assert "order id: 12345" in text


def test_easyocr_image_pdf(easyocr_engine: EasyOCREngine, image_only_pdf: Path) -> None:
    """Test that EasyOCR can process a PDF with only image content."""
    result = easyocr_engine.process_pdf(str(image_only_pdf))
    
    # Should use EasyOCR for processing
    assert result["engine"] == "easyocr"
    assert "pages" in result
    assert len(result["pages"]) == 1
    
    # Check content
    page = result["pages"][0]
    assert "text" in page
    assert "confidence" in page
    assert "boxes" in page
    
    # Verify some expected content from the invoice
    text = page["text"].lower()
    assert any(term in text for term in ["order", "invoice", "quantity", "total"])


def test_tesseract_selectable_pdf(tesseract_engine: TesseractEngine, selectable_pdf: Path) -> None:
    """Test that Tesseract can process a PDF with selectable text."""
    result = tesseract_engine.process_pdf(str(selectable_pdf))
    
    assert result["engine"] == "native_pdf"
    assert "pages" in result
    assert len(result["pages"]) == 2
    
    # Check content
    page1 = result["pages"][0]
    text = page1["text"].lower()
    assert "test pdf" in text
    assert "sample invoice" in text
    
    page2 = result["pages"][1]
    text = page2["text"].lower()
    assert "john doe" in text
    assert "order id: 12345" in text


def test_tesseract_image_pdf(tesseract_engine: TesseractEngine, image_only_pdf: Path) -> None:
    """Test that Tesseract can process a PDF with only image content."""
    result = tesseract_engine.process_pdf(str(image_only_pdf))
    
    assert result["engine"] == "tesseract"
    assert "pages" in result
    assert len(result["pages"]) == 1
    
    page = result["pages"][0]
    assert "text" in page
    assert "confidence" in page
    assert "boxes" in page
    
    text = page["text"].lower()
    assert any(term in text for term in ["order", "invoice", "quantity", "total"])


def test_ollama_selectable_pdf(ollama_engine: OllamaLLMEngine, selectable_pdf: Path) -> None:
    """Test that Ollama can process a PDF with selectable text."""
    result = ollama_engine.process_pdf(str(selectable_pdf))
    
    assert result["engine"] == "native_pdf"
    assert "pages" in result
    assert len(result["pages"]) == 2
    
    page1 = result["pages"][0]
    text = page1["text"].lower()
    assert "test pdf" in text
    assert "sample invoice" in text


def test_ollama_image_pdf(ollama_engine: OllamaLLMEngine, image_only_pdf: Path) -> None:
    """Test that Ollama can process a PDF with only image content."""
    result = ollama_engine.process_pdf(str(image_only_pdf))
    
    assert result["engine"] == "ollama"
    assert "pages" in result
    assert len(result["pages"]) == 1
    
    page = result["pages"][0]
    assert "text" in page
    assert "raw_response" in page
    
    text = page["text"].lower()
    assert any(term in text for term in ["order", "invoice", "quantity", "total"])


def test_gpt4_selectable_pdf(gpt4_engine: GPT4VisionEngine, selectable_pdf: Path) -> None:
    """Test that GPT-4 Vision can process a PDF with selectable text."""
    result = gpt4_engine.process_pdf(str(selectable_pdf))
    
    assert result["engine"] == "native_pdf"
    assert "pages" in result
    assert len(result["pages"]) == 2
    
    page1 = result["pages"][0]
    text = page1["text"].lower()
    assert "test pdf" in text
    assert "sample invoice" in text


def test_gpt4_image_pdf(gpt4_engine: GPT4VisionEngine, image_only_pdf: Path) -> None:
    """Test that GPT-4 Vision can process a PDF with only image content."""
    result = gpt4_engine.process_pdf(str(image_only_pdf))
    
    assert result["engine"] == "gpt4-vision"
    assert "pages" in result
    assert len(result["pages"]) == 1
    
    page = result["pages"][0]
    assert "text" in page
    assert "raw_response" in page
    
    text = page["text"].lower()
    assert any(term in text for term in ["order", "invoice", "quantity", "total"])


def test_surya_selectable_pdf(surya_engine: SuryaEngine, selectable_pdf: Path) -> None:
    """Test that Surya can process a PDF with selectable text."""
    result = surya_engine.process_pdf(str(selectable_pdf))
    
    assert result["engine"] == "native_pdf"
    assert "pages" in result
    assert len(result["pages"]) == 2
    
    page1 = result["pages"][0]
    text = page1["text"].lower()
    assert "test pdf" in text
    assert "sample invoice" in text


def test_surya_image_pdf(surya_engine: SuryaEngine, image_only_pdf: Path) -> None:
    """Test that Surya can process a PDF with only image content."""
    result = surya_engine.process_pdf(str(image_only_pdf))
    
    assert result["engine"] == "surya"
    assert "pages" in result
    assert len(result["pages"]) == 1
    
    page = result["pages"][0]
    assert "text" in page
    assert "confidence" in page
    assert "boxes" in page
    
    text = page["text"].lower()
    assert any(term in text for term in ["order", "invoice", "quantity", "total"])


def test_pdf_error_handling(
    easyocr_engine: EasyOCREngine,
    tesseract_engine: TesseractEngine,
    ollama_engine: OllamaLLMEngine,
    gpt4_engine: GPT4VisionEngine,
    surya_engine: SuryaEngine,
) -> None:
    """Test that all engines properly handle errors with invalid PDFs."""
    engines = [
        ("EasyOCR", easyocr_engine),
        ("Tesseract", tesseract_engine),
        ("Ollama", ollama_engine),
        ("GPT-4 Vision", gpt4_engine),
        ("Surya", surya_engine),
    ]
    
    non_existent_path = "/path/to/nonexistent.pdf"
    
    for engine_name, engine in engines:
        result = engine.process_pdf(non_existent_path)
        assert isinstance(result, dict)
        assert "error" in result, f"{engine_name} should return error for invalid PDF"
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0
