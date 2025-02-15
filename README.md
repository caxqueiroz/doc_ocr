# OCR Processing Application

This application provides both CLI and API interfaces for processing images and PDFs using various OCR engines.

## Features

- Multiple OCR engine support:
  - EasyOCR
  - Tesseract
  - (Placeholder for GPT4-Vision-Mini)
  - (Placeholder for Llama 3.2 Vision)
- Process both single files and directories
- Support for multiple languages
- FastAPI-based REST API
- Extensible architecture for future NER integration
- JSON output format

## Installation

1. Install system dependencies:
   ```bash
   # For Tesseract
   brew install tesseract
   
   # For PDF processing
   brew install poppler
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## CLI Usage

Process a single file:
```bash
python cli.py input.pdf output_dir --engines easyocr tesseract --languages en
```

Process a directory:
```bash
python cli.py input_dir output_dir --engines easyocr tesseract --recursive --languages en
```

## API Usage

Start the API server:
```bash
python api.py
```

The API will be available at `http://localhost:8000` with the following endpoints:

- `POST /process`: Process a single file
  ```bash
  curl -X POST "http://localhost:8000/process" \
       -H "accept: application/json" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@input.pdf" \
       -F "engines=easyocr,tesseract"
  ```

- `GET /health`: Health check endpoint

## Output Format

The OCR results are saved as JSON files with the following structure:
```json
{
  "EasyOCREngine": {
    "engine": "easyocr",
    "text": "extracted text",
    "confidence": [0.99, 0.98, ...],
    "boxes": [[x1, y1, x2, y2], ...]
  },
  "TesseractEngine": {
    "engine": "tesseract",
    "text": "extracted text",
    "confidence": [90, 85, ...],
    "boxes": [[left, top, width, height], ...]
  }
}
```

## Future Extensions

The codebase is designed to be modular and extensible. Future additions include:
- NER (Named Entity Recognition) processing
- Support for GPT4-Vision-Mini
- Support for Llama 3.2 Vision
- Additional OCR engines
- Batch processing optimization
