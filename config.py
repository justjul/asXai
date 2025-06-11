import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Core paths
CODE_ROOT = Path(os.getenv("CODE_ROOT", "/app")).resolve()
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/app")).resolve()
DRIVE_ROOT = (
    Path(os.getenv("DRIVE_ROOT",)).resolve()
    if os.getenv("DRIVE_ROOT",) else None
)
TEMP_ROOT = Path(os.getenv("TEMP_ROOT", "/app")).resolve()
VECTORDB_ROOT = Path(os.getenv("VECTORDB_ROOT", "/app/vectorDB")).resolve()

PROJECT_NAME = "asXai"

# qdrant docker server
# use container name by default
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
MODEL_EMBED = "intfloat/multilingual-e5-large-instruct"
VECTOR_SIZE = {"multilingual-e5-large-instruct": 1024}

# directory of the project
all_paths = []

# path to temporary folder
TMP_PATH = TEMP_ROOT
all_paths.append(TMP_PATH)

# path to data
DATA_PATH = PROJECT_ROOT / "data"
all_paths.append(DATA_PATH)

# path to metadata
METADATA_PATH = DATA_PATH / "metadata"
all_paths.append(METADATA_PATH)

# path to text data
TEXTDATA_PATH = DATA_PATH / "textdata"
all_paths.append(TEXTDATA_PATH)

# path to embeddings
VECTORDB_PATH = VECTORDB_ROOT
all_paths.append(VECTORDB_PATH)

DRIVE_DATA_PATH = DRIVE_ROOT / PROJECT_NAME / "data" if DRIVE_ROOT else None
all_paths.append(DRIVE_DATA_PATH)

DRIVE_TEXTDATA_PATH = DRIVE_DATA_PATH / "textdata" if DRIVE_ROOT else None
all_paths.append(DRIVE_TEXTDATA_PATH)

DRIVE_METADATA_PATH = DRIVE_DATA_PATH / "metadata" if DRIVE_ROOT else None
all_paths.append(DRIVE_METADATA_PATH)

# path to trained models
MODELS_PATH = PROJECT_ROOT / "models"
all_paths.append(MODELS_PATH)

LOGS_PATH = PROJECT_ROOT / "logs"
all_paths.append(LOGS_PATH)

USERS_ROOT = PROJECT_ROOT / "users"

# Auto-create folders on import
for path in all_paths:
    if path and not path.exists() and not path.is_symlink():
        path.mkdir(parents=True, exist_ok=True)
