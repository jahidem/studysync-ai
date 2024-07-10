import sys
import dotenv
dotenv.load_dotenv()
from studysync.utils import state
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from studysync.routers.api import api
import uvicorn
import logging
from rich.logging import RichHandler

# Setup Logger
rich_handler = RichHandler(rich_tracebacks=True)
rich_handler.setFormatter(fmt=logging.Formatter(fmt="%(message)s", datefmt="[%X]"))
logging.basicConfig(handlers=[rich_handler])

logger = logging.getLogger("studysync")


# Initialize the Application Server
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api, prefix="/api")


def run():

    # Load config from CLI
    state.cli_args = sys.argv[1:]

    if state.cli_args:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    logger.info("🌘 Starting StudySync")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
