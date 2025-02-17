from typing import Dict, Any, List
import spacy
from collections import defaultdict
import re
import logging
from .ocr_engines import (
    EasyOCREngine,
    PaddleOCREngine,
    TesseractEngine
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedOCRPipeline:
    def __init__(self):
        # Initialize OCR engines
        self.ocr_engines = {
            'easyocr': EasyOCREngine(languages=['en']),
            'paddleocr': PaddleOCREngine(languages='en'),
            'tesseract': TesseractEngine()
        }
        
        # Initialize NER model
        self.nlp = spacy.load("en_core_web_trf")
        
        # Compile regex patterns for validation
        self.patterns = {
            'email': re.compile(r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}'),  # More permissive email pattern
            'phone': re.compile(r'(?:\+\d{1,3}\s?)?(?:\(\d+\)|\d+)[-\s\d]{8,}'),  # International phone numbers
            'address': re.compile(
                r'.*(?:' + 
                '|'.join([
                    # Streets and roads
                    r'(?:road|rd|street|st|avenue|ave|boulevard|blvd|drive|dr|lane|ln|way)',
                    # Buildings and units
                    r'(?:building|bldg|floor|fl|suite|ste|unit|apt|apartment|room|rm)',
                    # Postal codes
                    r'(?:postal|zip|pin)\s*code',
                    # Unit numbers (common in Singapore)
                    r'#\d+[-\d]*'
                ]) + 
                r').*', 
                re.IGNORECASE
            )
        }

    def _combine_ocr_results(self, image_path: str) -> str:
        """Combine results from multiple OCR engines using a voting system"""
        all_texts = []
        confidences = []
        
        for engine_name, engine in self.ocr_engines.items():
            try:
                result = engine.process_image(image_path)
                if 'error' not in result:
                    text = result['text']
                    confidence = result.get('confidence', 0.5)  # Default confidence if not provided
                    all_texts.append(text)
                    confidences.append(confidence)
                    logger.info(f"{engine_name} confidence: {confidence}")
            except Exception as e:
                logger.error(f"Error with {engine_name}: {str(e)}")
        
        # If no results, return empty string
        if not all_texts:
            return ""
        
        # Use the result with highest confidence
        best_idx = confidences.index(max(confidences))
        return all_texts[best_idx]

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert common OCR errors
        replacements = {
            'l': '1',  # Common confusion between l and 1
            'O': '0',  # Common confusion between O and 0
            'S': '5',  # Common confusion between S and 5
            '|': 'I',  # Common confusion between | and I
        }
        
        # Only replace if in a numeric context
        cleaned = []
        for word in text.split():
            if word.isdigit():
                for old, new in replacements.items():
                    word = word.replace(old, new)
            cleaned.append(word)
        
        return ' '.join(cleaned)

    def _validate_entity(self, entity_type: str, value: str) -> bool:
        """Validate extracted entities"""
        if entity_type not in self.patterns:
            return True
            
        return bool(self.patterns[entity_type].match(value))

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract and validate entities from text"""
        print("Raw text for entity extraction:", text)  # Debug print
        doc = self.nlp(text)
        entities = defaultdict(list)
        
        # Extract entities using spaCy
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
                entities[ent.label_.lower()].append(ent.text)
        
        # Organization detection using common business keywords and patterns
        org_patterns = [
            # Look for company identifiers
            r'(?i)(?:ltd|limited|llc|inc|incorporated|corp|corporation)',
            # Look for business units/departments
            r'(?i)(?:sales|support|billing|shipping)\s+(?:department|division|unit)',
            # Look for business type indicators
            r'(?i)(?:store|shop|mall|market|enterprise|company)',
            # Look for product brands (all caps with optional spaces)
            r'(?:[A-Z][A-Z0-9]+(?:\s+[A-Z0-9]+)*)'  # e.g. SAMSUNG, APPLE INC, IBM
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                org_name = match.group().strip()
                # Clean up organization names
                org_name = org_name.strip()
                # Remove common noise words
                noise_words = ['the', 'a', 'an', 'and', 'or', 'of', 'to', 'for']
                org_words = [word for word in org_name.split() if word.lower() not in noise_words]
                org_name = ' '.join(org_words)
                
                # Keep original case for acronyms/brands, capitalize others
                if not org_name.isupper():
                    org_name = org_name.title()
                
                if org_name and org_name.lower() not in [x.lower() for x in entities['org']]:
                    entities['org'].append(org_name)
        
        # Extract structured entities using regex
        # First, try to find phone numbers in the entire text
        phone_matches = self.patterns['phone'].finditer(text)
        for match in phone_matches:
            phone = ''.join(filter(str.isdigit, match.group()))
            if len(phone) >= 8:  # Minimum length for valid phone
                entities['phone'].append(phone)
        
        # Then process line by line for other entities
        words = text.split('\n')
        for word in words:
            word = word.strip()
            if self.patterns['email'].match(word):
                entities['email'].append(word)
            elif self.patterns['address'].match(word):
                entities['address'].append(word)
        
        return dict(entities)

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image through the enhanced pipeline"""
        # Stage 1: Multi-Engine OCR
        raw_text = self._combine_ocr_results(image_path)
        if not raw_text:
            return {"error": "No text could be extracted from the image"}
        
        # Stage 2: Post-Processing
        cleaned_text = self._clean_text(raw_text)
        
        # Stage 3: Entity Recognition
        entities = self._extract_entities(cleaned_text)
        
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "entities": entities
        }
