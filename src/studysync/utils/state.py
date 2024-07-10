# Standard Packages
from typing import List, Dict
from studysync.processor.gemini import Gemini

# External Packages
from pathlib import Path

# Internal Packages
from studysync.utils.config import FileHandling, IndexContent, VectorDatabase, Generator


# Application Global State
config_file: Path = None
verbose: int = 0
host: str = None
port: int = None
cli_args: List[str] = None
previous_query: str = None
demo: bool = False
cli_args: List[str] = None

# global objs
file_handling = FileHandling()
vector_database = VectorDatabase()
gemini = Gemini()
index_content = IndexContent(gemini)
generator = Generator(vector_database, gemini)
