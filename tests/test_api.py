import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api import app

# Initialize test client
client = TestClient(app)


def test_process_invoice():
    """Test processing an invoice image through the API"""
    # Get the path to the test image
    image_path = Path(__file__).parent.parent / "data" / "invoice.jpeg"
    assert image_path.exists(), f"Test image not found at {image_path}"

    # Create the files payload
    with open(image_path, "rb") as f:
        files = {"file": ("invoice.jpeg", f, "image/jpeg")}
        response = client.post("/process", files=files)

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "results" in data
    assert data["filename"] == "invoice.jpeg"

    # Check that we got results from both OCR engines
    assert "EasyOCREngine" in data["results"]
    assert "TesseractEngine" in data["results"]

    # Check that text was extracted
    assert "text" in data["results"]["EasyOCREngine"]
    assert len(data["results"]["EasyOCREngine"]["text"]) > 0
    assert "text" in data["results"]["TesseractEngine"]
    assert len(data["results"]["TesseractEngine"]["text"]) > 0


def test_process_invalid_file():
    """Test uploading an invalid file type"""
    # Create a temporary text file
    test_file_content = b"This is not an image"
    files = {"file": ("test.txt", test_file_content, "text/plain")}
    response = client.post("/process", files=files)

    # Check response
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Unsupported file type" in data["detail"]


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
