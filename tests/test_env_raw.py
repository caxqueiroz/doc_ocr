import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_env_raw() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    logger.info(f"Reading .env from {env_path}")

    with open(env_path) as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip()
                logger.info(f"Found API key in file: {bool(key)}")
                logger.info(f"Key length: {len(key)}")
                break


if __name__ == "__main__":
    test_env_raw()
