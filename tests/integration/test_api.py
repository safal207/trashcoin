import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the root directory to the Python path to allow imports from 'api'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.app_fastapi import app

client = TestClient(app)

def test_classify_endpoint_success():
    """
    Tests the /api/classify endpoint with a valid image.
    """
    # This assumes the sample_images directory and the test image exist
    image_path = "sample_images/plastic_bottle.jpg"

    # Check if the sample image exists to avoid test errors
    if not os.path.exists(image_path):
        pytest.skip(f"Sample image not found at {image_path}, skipping test.")

    with open(image_path, "rb") as f:
        files = {"file": ("plastic_bottle.jpg", f, "image/jpeg")}
        response = client.post("/api/classify", files=files)

    assert response.status_code == 200
    data = response.json()

    # Check the response schema
    assert "class_" in data
    assert "confidence" in data
    assert "model_version" in data
    assert "inference_ms" in data

    # Check the types of the returned values
    assert isinstance(data["class_"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["model_version"], str)
    assert isinstance(data["inference_ms"], int)

def test_classify_endpoint_no_file():
    """
    Tests the endpoint response when no file is provided.
    """
    response = client.post("/api/classify")
    assert response.status_code == 422  # FastAPI uses 422 for validation errors

def test_classify_endpoint_wrong_file_type():
    """
    Tests the endpoint response when a non-image file is provided.
    """
    files = {"file": ("test.txt", "this is not an image", "text/plain")}
    response = client.post("/api/classify", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
