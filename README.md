# OCR and NER Processing Application

This application provides both CLI and API interfaces for processing images and PDFs using various OCR and NER engines. It supports text extraction and entity recognition using both cloud-based and local models.

## Features

- Multiple OCR engine support:
  - EasyOCR (local)
  - Tesseract (local)
  - OpenAI GPT-4 Vision (cloud)
  - Ollama Llama 3.2 Vision (local)

- Named Entity Recognition (NER) support:
  - OpenAI GPT-4 (cloud)
  - Ollama Llama 3.2 (local)
  - SpaCy (local)

- Process both single files and directories
- Support for multiple languages
- FastAPI-based REST API
- Modular architecture for easy engine integration
- Structured JSON output format

## Installation

1. Install system dependencies:
   ```bash
   # For Tesseract
   brew install tesseract
   
   # For PDF processing
   brew install poppler

   # For Ollama (local LLM support)
   brew install ollama
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Pull required Ollama models:
   ```bash
   ollama pull llama3.2-vision:latest
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

The application provides both OCR and NER results in structured JSON format:

### OCR Output
```json
{
  "EasyOCREngine": {
    "engine": "easyocr",
    "text": "extracted text",
    "confidence": [0.99, 0.98, ...],
    "boxes": [[x1, y1, x2, y2], ...]
  },
  "OllamaLLMEngine": {
    "engine": "llama3.2-vision",
    "text": "extracted text"
  },
  "OpenAIVisionEngine": {
    "engine": "gpt-4-vision",
    "text": "extracted text"
  }
}
```

### NER Output
```json
{
  "OpenAINEREngine": {
    "entities": {
      "person_name": ["John Doe"],
      "organization": "Acme Corp",
      "location": ["123 Main St, City"],
      "contact_info": {
        "phone": "+1234567890",
        "email": "contact@example.com"
      }
    }
  },
  "OllamaNEREngine": {
    "entities": {
      "order_number": "#12345",
      "date": "2025-02-15",
      "person_name": ["John Doe"],
      "organization": "Acme Corp"
    }
  }
}
```

## Environment Variables

Configure the following in your `.env` file:

```bash
# OCR Configuration
TESSERACT_PATH=/usr/local/bin/tesseract
EASYOCR_CACHE_DIR=~/.cache/easyocr

# LLM Models Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama3.2-vision:latest
OLLAMA_NER_MODEL=llama3.2-vision:latest

# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4-vision-preview
```

## Future Extensions

The codebase is designed to be modular and extensible. Future additions include:
- Support for additional OCR engines
- Integration with more LLM models
- Batch processing optimization
- Custom model training support
- Enhanced entity extraction capabilities
