from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import tempfile
import shutil
import logging
from typing import List, Optional
from pydantic import BaseModel
from ocr_processor.ocr_engines import EasyOCREngine, TesseractEngine
from ocr_processor.processor import OCRProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OCR Processing API",
    description="API for processing images and PDFs using various OCR engines",
)

# Initialize OCR engines (you might want to make this configurable)
engines = [EasyOCREngine(languages=["en"]), TesseractEngine(lang="eng")]


class OCRResponse(BaseModel):
    filename: str
    results: dict


@app.post("/process", response_model=OCRResponse)
async def process_file(
    file: UploadFile = File(...),
    engines: Optional[List[str]] = ["easyocr", "tesseract"],
):
    """
    Process a single file with specified OCR engines
    """
    # Validate file type
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            ),
        )

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / file.filename

        # Save uploaded file
        try:
            with temp_file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

        # Initialize processor with temporary output directory
        processor = OCRProcessor(engines=engines, output_dir=temp_dir)

        try:
            results = processor.process_file(str(temp_file_path))
            return OCRResponse(filename=file.filename, results=results)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing file: {str(e)}"
            )


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
