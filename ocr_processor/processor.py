import os
from typing import List, Dict, Any
import json
from pathlib import Path
import logging
from .ocr_engines import OCREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRProcessor:
    def __init__(self, engines: List[OCREngine], output_dir: str) -> None:
        self.engines = engines
        self.output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def process_file(self, file_path: str | Path) -> Dict[str, Any]:
        """Process a single file with all configured engines"""
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        results = {}

        if not file_path.exists():
            return {"error": f"File {file_path} does not exist"}

        file_extension = file_path.suffix.lower()

        for engine in self.engines:
            try:
                if file_extension in [".pdf"]:
                    result = engine.process_pdf(str(file_path))
                elif file_extension in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
                    result = engine.process_image(str(file_path))
                else:
                    continue

                results[engine.__class__.__name__] = result
            except Exception as e:
                logger.error(
                    f"Error processing {file_path} with {engine.__class__.__name__}: {str(e)}"
                )
                results[engine.__class__.__name__] = {"error": str(e)}

        return results

    def process_directory(
        self, input_dir: str | Path, recursive: bool = True
    ) -> Dict[str, Any]:
        """Process all files in a directory"""
        input_dir = Path(input_dir) if isinstance(input_dir, str) else input_dir
        if not input_dir.exists():
            return {"error": f"Directory {input_dir} does not exist"}

        processed_files = {}
        pattern = "**/*" if recursive else "*"

        for file_path in input_dir.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in [
                ".pdf",
                ".png",
                ".jpg",
                ".jpeg",
                ".tiff",
                ".gif",
            ]:
                logger.info(f"Processing {file_path}")
                results = self.process_file(str(file_path))

                # Save results to JSON file
                relative_path = file_path.relative_to(input_dir)
                output_path = Path(self.output_dir) / f"{relative_path.stem}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                processed_files[str(relative_path)] = results

        return processed_files


class NERProcessor:
    """Placeholder for future NER processing capabilities"""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        # TODO: Initialize NER model

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text with NER model

        Args:
            text (str): Text to process

        Returns:
            dict: Named entities extracted from text
        """
        raise NotImplementedError("NER processing not yet implemented")
