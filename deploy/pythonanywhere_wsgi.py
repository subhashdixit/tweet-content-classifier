"""Example WSGI configuration: set PROJECT_ROOT to the actual account checkout."""

# Dynamic import follows provider path configuration.
# pylint: disable=wrong-import-position,import-error

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(
    os.environ.get("TWEET_PROJECT_ROOT", "/home/YOUR_USERNAME/tweet-content-classifier")
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ["MODEL_DIR"] = str(PROJECT_ROOT / "models")
revision_file = PROJECT_ROOT / "RELEASE_REVISION"
os.environ["APP_REVISION"] = (
    revision_file.read_text().strip()
    if revision_file.exists()
    else "manual-unversioned"
)
from tweet_classifier.web import (
    create_app,
)  # pylint: disable=wrong-import-position,import-error

application = create_app()
