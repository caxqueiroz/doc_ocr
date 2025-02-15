import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from ner_processor.ner_engines import OpenAINEREngine, LlamaNEREngine, SpacyNEREngine

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
with open(env_path) as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY='):
            key = line.split('=', 1)[1].strip()
            os.environ['OPENAI_API_KEY'] = key
            break

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ner_plain_text():
    # Initialize the NER engines
    engines = [
        OpenAINEREngine(),
        LlamaNEREngine(),
        SpacyNEREngine()
    ]

    # Test text
    text = """
    John Smith from Apple Inc. visited our office in New York on January 15, 2025.
    He discussed purchasing 50 MacBook Pro laptops at $2,000 each for the engineering team.
    You can reach him at john.smith@apple.com or +1-555-0123.
    """

    logger.info("Processing plain text:")
    logger.info("-" * 50)
    logger.info(text)
    logger.info("-" * 50)

    # Process the text with each engine
    results = []
    for engine in engines:
        logger.info(f"\nProcessing with {engine.__class__.__name__}:")
        logger.info("-" * 50)
        result = engine.process_text(text)
        results.append(result)

        # Log the result
        if "error" in result:
            logger.error(f"Error: {result['error']}")
        else:
            logger.info(f"Entities extracted: {json.dumps(result['entities'], indent=2)}")
            logger.info(f"Model used: {result['model']}")
        logger.info("-" * 50)

    return results

def test_ner_json_schema():
    # Initialize the NER engines
    engines = [
        OpenAINEREngine(),
        LlamaNEREngine(),
        SpacyNEREngine()
    ]

    # Test JSON schema
    json_text = """{
        "order_details": {
            "customer": {
                "name": "Jane Doe",
                "company": "Tech Solutions Ltd",
                "email": "jane.doe@techsolutions.com"
            },
            "products": [
                {
                    "name": "Dell XPS 13",
                    "quantity": 5,
                    "price": 1299.99
                },
                {
                    "name": "HP Monitor 27 inch",
                    "quantity": 10,
                    "price": 299.99
                }
            ],
            "shipping_address": {
                "street": "123 Business Ave",
                "city": "San Francisco",
                "state": "CA",
                "zip": "94105"
            },
            "order_date": "2025-02-15"
        }
    }"""

    logger.info("Processing JSON schema:")
    logger.info("-" * 50)
    logger.info(json_text)
    logger.info("-" * 50)

    # Process the JSON with each engine
    results = []
    for engine in engines:
        logger.info(f"\nProcessing with {engine.__class__.__name__}:")
        logger.info("-" * 50)
        result = engine.process_json_schema(json_text)
        results.append(result)

        # Log the result
        if "error" in result:
            logger.error(f"Error: {result['error']}")
        else:
            logger.info(f"Entities extracted: {json.dumps(result['entities'], indent=2)}")
            logger.info(f"Model used: {result['model']}")
        logger.info("-" * 50)

    return results

if __name__ == "__main__":
    test_ner_plain_text()
    test_ner_json_schema()
