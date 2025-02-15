import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent

    # OCR Engines
    TESSERACT_PATH: str = os.getenv("TESSERACT_PATH", "/usr/local/bin/tesseract")
    EASYOCR_CACHE_DIR: str = os.getenv("EASYOCR_CACHE_DIR", "~/.cache/easyocr")

    # LLM Models
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_DEFAULT_MODEL: str = os.getenv("OLLAMA_DEFAULT_MODEL", "llava")

    # OpenAI (if needed)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-vision-preview")

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/ocr_processor.log")

    # Output
    DEFAULT_OUTPUT_DIR: Path = Path(os.getenv("DEFAULT_OUTPUT_DIR", "./output"))
    TEMP_DIR: Path = Path(os.getenv("TEMP_DIR", "/tmp/ocr_processor"))

    @classmethod
    def setup(cls):
        """Setup necessary directories and configurations"""
        # Create output directory
        cls.DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Create logs directory
        Path(cls.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

        # Create temp directory
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Expand user path for EasyOCR cache
        cls.EASYOCR_CACHE_DIR = str(Path(cls.EASYOCR_CACHE_DIR).expanduser())


# Initialize configuration
config = Config()
config.setup()
