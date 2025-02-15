import argparse
import logging
from pathlib import Path
from typing import List
from ocr_processor.ocr_engines import (
    OCREngine,
    EasyOCREngine,
    TesseractEngine,
    GPT4VisionEngine,
    OllamaLLMEngine,
)
from ocr_processor.processor import OCRProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR Processing CLI")
    parser.add_argument("input_path", type=str, help="Input file or directory path")
    parser.add_argument("output_dir", type=str, help="Output directory for results")
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["easyocr", "tesseract"],
        choices=["easyocr", "tesseract", "gpt4-vision-mini", "llama-vision"],
        help="OCR engines to use",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Process directories recursively"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Languages to process (ISO 639-1 codes)",
    )

    args = parser.parse_args()

    # Initialize selected engines
    engines: List[OCREngine] = []
    for engine in args.engines:
        if engine == "easyocr":
            engines.append(EasyOCREngine(languages=args.languages))
        elif engine == "tesseract":
            engines.append(TesseractEngine(lang="+".join(args.languages)))
        elif engine == "gpt4-vision-mini":
            engines.append(GPT4VisionEngine())
        elif engine == "llama-vision":
            engines.append(OllamaLLMEngine("llama-3.2-vision"))

    processor = OCRProcessor(engines=engines, output_dir=args.output_dir)

    input_path = Path(args.input_path)
    if input_path.is_file():
        logger.info(f"Processing single file: {input_path}")
        processor.process_file(str(input_path))
        logger.info(f"Results saved to {args.output_dir}")
    elif input_path.is_dir():
        logger.info(f"Processing directory: {input_path}")
        processor.process_directory(str(input_path), recursive=args.recursive)
        logger.info(f"Results saved to {args.output_dir}")
    else:
        logger.error(f"Input path {input_path} does not exist")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
