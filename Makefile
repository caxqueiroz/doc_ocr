.PHONY: help install run-api run-cli test lint clean setup-dev setup-ollama

# Default target when just running 'make'
help:
	@echo "Available commands:"
	@echo "  make install       - Install all dependencies"
	@echo "  make setup-dev     - Setup development environment"
	@echo "  make setup-ollama  - Setup Ollama and pull required models"
	@echo "  make run-api       - Run the FastAPI server"
	@echo "  make run-cli       - Run the CLI (requires INPUT and OUTPUT args)"
	@echo "  make test          - Run tests"
	@echo "  make test-ocr      - Run OCR-specific tests"
	@echo "  make test-ner      - Run NER-specific tests"
	@echo "  make lint          - Run linting"
	@echo "  make clean         - Clean up temporary files"

# Install dependencies
install:
	pip install -r requirements.txt

# Setup development environment with additional tools
setup-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

# Setup Ollama and pull required models
setup-ollama:
	@echo "Checking if Ollama is installed..."
	@if ! command -v ollama >/dev/null 2>&1; then \
		echo "Installing Ollama..."; \
		brew install ollama; \
	fi
	@echo "Pulling required Ollama models..."
	@ollama pull llama3.2-vision:latest

# Run the API server
run-api:
	python api.py

# Run the CLI
# Usage: make run-cli INPUT=path/to/input OUTPUT=path/to/output [ENGINES="easyocr tesseract"] [RECURSIVE=true]
run-cli:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Error: INPUT and OUTPUT are required."; \
		echo "Usage: make run-cli INPUT=path/to/input OUTPUT=path/to/output [ENGINES=\"easyocr tesseract\"] [RECURSIVE=true]"; \
		exit 1; \
	fi
	python cli.py $(INPUT) $(OUTPUT) \
		$(if $(ENGINES),--engines $(ENGINES),) \
		$(if $(RECURSIVE),--recursive,)

# Run tests
test:
	pytest tests/ --cov=ocr_processor --cov=ner_processor -v

# Run OCR-specific tests
test-ocr:
	pytest tests/test_ocr*.py -v

# Run NER-specific tests
test-ner:
	pytest tests/test_ner*.py -v

# Run linting
lint:
	black .
	flake8 .
	mypy .

# Clean up temporary files and artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name "build" -exec rm -r {} +
	find . -type d -name "dist" -exec rm -r {} +
