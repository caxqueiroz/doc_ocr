from abc import ABC, abstractmethod
from typing import Dict, Any

class OCREngine(ABC):
    @abstractmethod
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process an image and extract text

        Args:
            image_path (str): Path to the image file

        Returns:
            dict: JSON schema with extracted text
        """
        raise NotImplementedError

    @abstractmethod
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF and extract text

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            dict: JSON schema with extracted text
        """
        raise NotImplementedError
