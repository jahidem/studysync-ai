# Standard Packages
from typing import List, Dict


# External Packages
from pathlib import Path

# Internal Packages
from studysync.utils.config import FileHandling, IndexContent, VectorDatabase


# Application Global State
config_file: Path = None
verbose: int = 0
host: str = None
port: int = None
cli_args: List[str] = None
previous_query: str = None
demo: bool = False

# global objs
file_handling = FileHandling()
vector_database = VectorDatabase()
index_content = IndexContent()