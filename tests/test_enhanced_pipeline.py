from pathlib import Path
import pytest
from ocr_processor.enhanced_ocr_pipeline import EnhancedOCRPipeline


@pytest.fixture
def invoice_path() -> Path:
    """Fixture to provide the path to the test invoice image"""
    path = Path(__file__).parent.parent / "data" / "invoice.jpeg"
    assert path.exists(), f"Test invoice not found at {path}"
    return path


@pytest.fixture
def pipeline() -> EnhancedOCRPipeline:
    """Fixture to provide a configured EnhancedOCRPipeline instance"""
    return EnhancedOCRPipeline()


def test_enhanced_pipeline_entity_extraction(
    pipeline: EnhancedOCRPipeline, invoice_path: Path
) -> None:
    """Test that the enhanced pipeline can extract entities accurately"""
    # Process the invoice image
    result = pipeline.process_image(str(invoice_path))

    # Check basic structure of the result
    assert isinstance(result, dict)
    assert "raw_text" in result
    assert "cleaned_text" in result
    assert "entities" in result

    # Check entities
    entities = result["entities"]
    
    # 1. Person Name
    assert "person" in entities, "No person entities found"
    assert any("carlos queiroz" in name.lower() for name in entities["person"]), \
        "Customer name not found in person entities"
    
    # 2. Address
    assert "address" in entities, "No address entities found"
    assert any("kellock road" in addr.lower() for addr in entities["address"]), \
        "Address not found in address entities"
    
    # 3. Phone Number
    assert "phone" in entities, "No phone entities found"
    assert any("91937205" in phone for phone in entities["phone"]), \
        "Phone number not found in phone entities"
    
    # 4. Organization
    assert "org" in entities, "No organization entities found"
    # Check that we found at least one organization
    assert len(entities["org"]) > 0, "No organizations detected"
    # Verify organization names are properly formatted
    for org in entities["org"]:
        # Organizations should either be all caps (brands) or title case
        assert org.isupper() or org == org.title(), \
            f"Organization {org} not properly formatted"
    
    # 5. Location
    assert "gpe" in entities, "No location entities found"
    assert any("singapore" in loc.lower() for loc in entities["gpe"]), \
        "Location not found in location entities"


def test_enhanced_pipeline_text_cleaning(
    pipeline: EnhancedOCRPipeline, invoice_path: Path
) -> None:
    """Test that the enhanced pipeline cleans text effectively"""
    # Process the invoice image
    result = pipeline.process_image(str(invoice_path))
    
    # Check cleaned text
    cleaned_text = result["cleaned_text"]
    
    # Test number corrections
    assert "85714" in cleaned_text, "Order number not properly cleaned"
    
    # Test dimension corrections (if OCR mistakes l for 1 or O for 0)
    dimension_terms = ["W1200", "L1900"]
    for term in dimension_terms:
        # Convert both to same case for comparison
        assert term.lower() in cleaned_text.lower(), f"Dimension {term} not properly cleaned"


def test_enhanced_pipeline_error_handling(pipeline: EnhancedOCRPipeline) -> None:
    """Test that the enhanced pipeline handles errors gracefully"""
    # Test with non-existent file
    result = pipeline.process_image("nonexistent.jpg")
    assert "error" in result
    assert isinstance(result["error"], str)

    # Test with empty string
    result = pipeline.process_image("")
    assert "error" in result
    assert isinstance(result["error"], str)
