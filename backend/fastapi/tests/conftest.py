import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure backend/fastapi root is on python import path for tests
fastapi_root = Path(__file__).parent.parent
if str(fastapi_root) not in sys.path:
    sys.path.insert(0, str(fastapi_root))

from main import app

@pytest.fixture
def client():
    return TestClient(app)
