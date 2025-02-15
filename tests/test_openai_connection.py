import logging
from openai import OpenAI
from ocr_processor.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_openai_connection() -> bool:
    """Test basic OpenAI API connectivity"""
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        models = client.models.list()
        logger.info("Successfully connected to OpenAI API")
        logger.info(f"Available models: {[model.id for model in models.data[:5]]}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to OpenAI API: {str(e)}")
        return False


if __name__ == "__main__":
    test_openai_connection()
