import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_env():
    # Try to load .env file
    env_path = Path(__file__).parent.parent / '.env'
    logger.info(f'Loading .env from {env_path}')
    load_dotenv(env_path)
    
    # Print environment variables
    logger.info(f'OPENAI_API_KEY: {os.getenv("OPENAI_API_KEY")}')
    logger.info(f'OPENAI_MODEL: {os.getenv("OPENAI_MODEL")}')
    logger.info(f'OPENAI_BASE_URL: {os.getenv("OPENAI_BASE_URL")}')

if __name__ == "__main__":
    test_env()
