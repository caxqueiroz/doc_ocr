from typing import List, Optional, Any
from PIL.Image import Image

class TextLine:
    text: str
    confidence: float
    bbox: List[float]

class RecognitionResult:
    text_lines: List[TextLine]

class RecognitionPredictor:
    def __init__(self) -> None: ...
    def __call__(
        self, images: List[Image], langs: List[Optional[str]], detection_predictor: Any
    ) -> List[RecognitionResult]: ...
