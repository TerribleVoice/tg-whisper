import os
import logging
import uvicorn
from app import app
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("whisper-server")

if __name__ == "__main__":
    port = os.getenv("PORT")
    if not port:
        logger.error("PORT is not set")
        raise ValueError("PORT is not set")
    port = int(port)
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Запуск сервера на {host}:{port}")
    uvicorn.run(app, host=host, port=port)
