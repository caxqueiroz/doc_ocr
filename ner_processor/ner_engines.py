import json
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import spacy
from openai import OpenAI

from ocr_processor.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseNEREngine(ABC):
    """Base class for all NER engines"""

    @abstractmethod
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process plain text and extract entities

        Args:
            text (str): Plain text to process

        Returns:
            Dict[str, Any]: JSON schema with extracted entities
        """
        pass

    @abstractmethod
    def process_json_schema(self, json_text: str) -> Dict[str, Any]:
        """Process text in JSON schema format and extract entities

        Args:
            json_text (str): Text in JSON format to process

        Returns:
            Dict[str, Any]: JSON schema with extracted entities
        """
        pass

    def _flatten_json(self, obj: Any, parent_key: str = "", sep: str = " ") -> str:
        """Flatten a nested JSON object into a single string

        Args:
            obj: JSON object to flatten
            parent_key: Key from parent object
            sep: Separator between key-value pairs

        Returns:
            str: Flattened text representation
        """
        items: List[str] = []
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.append(self._flatten_json(v, new_key, sep))
                else:
                    items.append(f"{new_key}: {v}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}[{i}]" if parent_key else str(i)
                if isinstance(v, (dict, list)):
                    items.append(self._flatten_json(v, new_key, sep))
                else:
                    items.append(f"{new_key}: {v}")
        else:
            items.append(str(obj))

        return sep.join(items)

class OpenAINEREngine(BaseNEREngine):
    """Named Entity Recognition engine using GPT-4 for entity extraction"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY
        )
        self.model = config.OPENAI_MODEL

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process plain text and extract entities

        Args:
            text (str): Plain text to process

        Returns:
            Dict[str, Any]: JSON schema with extracted entities
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Named Entity Recognition expert. Extract all entities from the text and format them in a JSON schema. Include categories like: person_name, organization, location, date, contact_info, product, quantity, price, and any other relevant entities found in the text."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Extract JSON from response
            result_text = response.choices[0].message.content
            try:
                # Try to parse as pure JSON
                entities = json.loads(result_text)
            except json.JSONDecodeError:
                # If not pure JSON, try to extract JSON from markdown
                if "```json" in result_text:
                    json_text = result_text.split("```json")[1].split("```")[0].strip()
                    entities = json.loads(json_text)
                else:
                    raise ValueError("Could not parse JSON from response")

            return {
                "engine": "gpt4-ner",
                "model": self.model,
                "entities": entities,
                "raw_response": response.model_dump()
            }

        except Exception as e:
            logger.error(f"Error processing text with NER: {str(e)}")
            return {"error": str(e)}

    def process_json_schema(self, json_text: str) -> Dict[str, Any]:
        """Process text in JSON schema format and extract entities

        Args:
            json_text (str): Text in JSON format to process

        Returns:
            Dict[str, Any]: JSON schema with extracted entities
        """
        try:
            # First validate and parse the input JSON
            try:
                input_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON input: {str(e)}")

            # Convert JSON to a more natural text format
            flattened_text = self._flatten_json(input_data)

            # Process the flattened text
            return self.process_text(flattened_text)

        except Exception as e:
            logger.error(f"Error processing JSON schema with NER: {str(e)}")
            return {"error": str(e)}

class OllamaNEREngine(BaseNEREngine):
    """Named Entity Recognition engine using local Ollama model"""

    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.OLLAMA_NER_MODEL

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process plain text and extract entities using local Ollama model"""
        try:
            prompt = (
                "You are a Named Entity Recognition expert. Extract all entities from the text below and format them in a valid JSON object. "
                "Follow these rules strictly:\n"
                "1. Use double quotes for all property names and string values\n"
                "2. Ensure all JSON syntax is valid\n"
                "3. Keep the structure flat and simple\n"
                "4. Group similar entities together\n\n"
                "Extract these types of entities:\n"
                "- order_number: Order reference numbers\n"
                "- date: Any dates or time periods\n"
                "- person_name: Names of people\n"
                "- organization: Company or business names\n"
                "- location: Addresses, cities, countries\n"
                "- contact_info: Phone numbers, emails, websites\n"
                "- product: Product names and descriptions\n"
                "- quantity: Numerical quantities\n"
                "- price: Monetary values and prices\n\n"
                f"Text: {text}\n\n"
                "Respond with a valid JSON object containing the extracted entities. Only output the JSON, nothing else."
            )


            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1000
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            result_text = result.get("response", "")

            try:
                # Try to parse as pure JSON
                entities = json.loads(result_text)
            except json.JSONDecodeError:
                # If not pure JSON, try to extract JSON from markdown
                if "```json" in result_text:
                    json_text = result_text.split("```json")[1].split("```")[0].strip()
                    entities = json.loads(json_text)
                else:
                    raise ValueError("Could not parse JSON from response")

            return {
                "engine": "ollama-ner",
                "model": self.model,
                "entities": entities,
                "raw_response": result
            }

        except Exception as e:
            logger.error(f"Error processing text with Ollama NER: {str(e)}")
            return {"error": str(e)}

    def process_json_schema(self, json_text: str) -> Dict[str, Any]:
        """Process text in JSON schema format using Ollama"""
        try:
            # First validate and parse the input JSON
            try:
                input_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON input: {str(e)}")

            # Convert JSON to a more natural text format
            flattened_text = self._flatten_json(input_data)

            # Process the flattened text
            return self.process_text(flattened_text)

        except Exception as e:
            logger.error(f"Error processing JSON schema with Llama NER: {str(e)}")
            return {"error": str(e)}

class SpacyNEREngine(BaseNEREngine):
    """Named Entity Recognition engine using spaCy"""

    def __init__(self):
        # Load English language model
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except OSError:
            # If transformer model not found, try loading the large model
            try:
                self.nlp = spacy.load("en_core_web_lg")
            except OSError:
                # If large model not found, download and load it
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
                self.nlp = spacy.load("en_core_web_lg")

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process plain text and extract entities using spaCy"""
        try:
            # Process the text
            doc = self.nlp(text)

            # Initialize entity categories
            entities = {
                "person": [],
                "organization": [],
                "location": [],
                "date": [],
                "money": [],
                "product": [],
                "quantity": [],
                "contact_info": []
            }

            # Extract entities
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "PER"]:
                    entities["person"].append(ent.text)
                elif ent.label_ in ["ORG", "ORGANIZATION"]:
                    entities["organization"].append(ent.text)
                elif ent.label_ in ["GPE", "LOC", "LOCATION"]:
                    entities["location"].append(ent.text)
                elif ent.label_ in ["DATE", "TIME"]:
                    entities["date"].append(ent.text)
                elif ent.label_ == "MONEY":
                    entities["money"].append(ent.text)
                elif ent.label_ == "PRODUCT":
                    entities["product"].append(ent.text)
                elif ent.label_ in ["QUANTITY", "CARDINAL"]:
                    entities["quantity"].append(ent.text)

            # Extract email addresses and phone numbers using pattern matching
            for token in doc:
                if token.like_email:
                    entities["contact_info"].append({"type": "email", "value": token.text})
                elif token.like_num and len(token.text) >= 10:
                    entities["contact_info"].append({"type": "phone", "value": token.text})

            # Remove empty categories
            entities = {k: v for k, v in entities.items() if v}

            return {
                "engine": "spacy-ner",
                "model": self.nlp.meta["name"],
                "entities": entities
            }

        except Exception as e:
            logger.error(f"Error processing text with spaCy NER: {str(e)}")
            return {"error": str(e)}

    def process_json_schema(self, json_text: str) -> Dict[str, Any]:
        """Process text in JSON schema format using spaCy"""
        try:
            # First validate and parse the input JSON
            try:
                input_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON input: {str(e)}")

            # Convert JSON to a more natural text format
            flattened_text = self._flatten_json(input_data)

            # Process the flattened text
            return self.process_text(flattened_text)

        except Exception as e:
            logger.error(f"Error processing JSON schema with spaCy NER: {str(e)}")
            return {"error": str(e)}
