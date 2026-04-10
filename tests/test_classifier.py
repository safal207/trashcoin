"""
Tests for classifier.py and Flask /classify endpoint.

Run with:
    pip install pytest
    pytest tests/ -v
"""

import io
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(color=(128, 64, 32), size=(224, 224)) -> bytes:
    """Return in-memory JPEG bytes for a solid-colour image."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _write_tmp_image(color=(128, 64, 32)) -> str:
    """Write a temporary JPEG and return its path."""
    data = _make_jpeg_bytes(color)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# classifier.py — unit tests
# ---------------------------------------------------------------------------

class TestClassifyWithLocalModel:
    def test_returns_known_class(self):
        from classifier import TRASH_CLASSES, classify_with_local_model

        # Build a fake model whose predict returns a one-hot vector
        mock_model = MagicMock()
        num_classes = len(TRASH_CLASSES)
        # Predict "Plastic" (index depends on loaded class list)
        target_idx = TRASH_CLASSES.index(
            next(c for c in TRASH_CLASSES if c.lower() == "plastic")
        )
        probs = np.zeros((1, num_classes))
        probs[0, target_idx] = 1.0
        mock_model.predict.return_value = probs

        img_path = _write_tmp_image()
        try:
            result = classify_with_local_model(img_path, mock_model)
        finally:
            os.unlink(img_path)

        assert result == TRASH_CLASSES[target_idx]

    def test_all_classes_reachable(self):
        """Verify that argmax for every index maps back to a class string."""
        from classifier import TRASH_CLASSES, classify_with_local_model

        mock_model = MagicMock()
        num_classes = len(TRASH_CLASSES)
        img_path = _write_tmp_image()
        try:
            for idx in range(num_classes):
                probs = np.zeros((1, num_classes))
                probs[0, idx] = 1.0
                mock_model.predict.return_value = probs
                result = classify_with_local_model(img_path, mock_model)
                assert result == TRASH_CLASSES[idx]
        finally:
            os.unlink(img_path)


class TestClassifyWithExternalApi:
    def test_raises_when_url_not_configured(self):
        from classifier import ExternalClassifierConfig, classify_with_external_api

        cfg = ExternalClassifierConfig(api_url=None, api_token=None)
        with pytest.raises(RuntimeError, match="EXTERNAL_CLASSIFIER_API_URL"):
            classify_with_external_api("/fake/path.jpg", cfg)

    def test_raises_on_http_error(self):
        from classifier import ExternalClassifierConfig, classify_with_external_api

        cfg = ExternalClassifierConfig(api_url="http://example.com/classify", api_token=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        img_path = _write_tmp_image()
        try:
            with patch("classifier.requests.post", return_value=mock_resp):
                with pytest.raises(RuntimeError, match="500"):
                    classify_with_external_api(img_path, cfg)
        finally:
            os.unlink(img_path)

    def test_returns_classification_field(self):
        from classifier import ExternalClassifierConfig, classify_with_external_api

        cfg = ExternalClassifierConfig(api_url="http://example.com/classify", api_token=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"classification": "Glass"}

        img_path = _write_tmp_image()
        try:
            with patch("classifier.requests.post", return_value=mock_resp):
                result = classify_with_external_api(img_path, cfg)
        finally:
            os.unlink(img_path)

        assert result == "Glass"

    def test_returns_label_field_as_fallback(self):
        from classifier import ExternalClassifierConfig, classify_with_external_api

        cfg = ExternalClassifierConfig(api_url="http://example.com/classify", api_token=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"label": "Metal"}

        img_path = _write_tmp_image()
        try:
            with patch("classifier.requests.post", return_value=mock_resp):
                result = classify_with_external_api(img_path, cfg)
        finally:
            os.unlink(img_path)

        assert result == "Metal"

    def test_raises_on_invalid_payload(self):
        from classifier import ExternalClassifierConfig, classify_with_external_api

        cfg = ExternalClassifierConfig(api_url="http://example.com/classify", api_token=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "something"}

        img_path = _write_tmp_image()
        try:
            with patch("classifier.requests.post", return_value=mock_resp):
                with pytest.raises(RuntimeError, match="invalid payload"):
                    classify_with_external_api(img_path, cfg)
        finally:
            os.unlink(img_path)

    def test_sends_auth_header_when_token_present(self):
        from classifier import ExternalClassifierConfig, classify_with_external_api

        cfg = ExternalClassifierConfig(
            api_url="http://example.com/classify", api_token="secret-token"
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"classification": "Paper"}

        img_path = _write_tmp_image()
        try:
            with patch("classifier.requests.post", return_value=mock_resp) as mock_post:
                classify_with_external_api(img_path, cfg)
                _, kwargs = mock_post.call_args
                assert kwargs["headers"].get("Authorization") == "Bearer secret-token"
        finally:
            os.unlink(img_path)


class TestClassifyTrashRouting:
    def test_uses_local_model_when_available(self):
        from classifier import TRASH_CLASSES

        mock_model = MagicMock()
        probs = np.zeros((1, len(TRASH_CLASSES)))
        probs[0, 0] = 1.0
        mock_model.predict.return_value = probs

        img_path = _write_tmp_image()
        try:
            with patch("classifier.TRASHNET_MODEL", mock_model):
                from classifier import classify_trash
                result = classify_trash(img_path, mock_model)
        finally:
            os.unlink(img_path)

        assert result in TRASH_CLASSES

    def test_falls_back_to_external_when_model_none(self):
        img_path = _write_tmp_image()
        try:
            with patch("classifier.classify_with_external_api", return_value="Trash") as mock_ext:
                from classifier import classify_trash
                result = classify_trash(img_path, model=None)
                mock_ext.assert_called_once()
        finally:
            os.unlink(img_path)

        assert result == "Trash"


# ---------------------------------------------------------------------------
# app.py — Flask endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Flask test client with classification mocked out and rate limiter disabled."""
    with patch("classifier.TRASHNET_MODEL", None):
        with patch("classifier.load_trashnet_model", return_value=None):
            import importlib
            import app as app_module
            importlib.reload(app_module)
            app_module.app.config["TESTING"] = True
            # Disable rate limiting in tests
            app_module.limiter.enabled = False
            with app_module.app.test_client() as c:
                yield c
            app_module.limiter.enabled = True


class TestClassifyEndpoint:
    def test_no_file_returns_400(self, client):
        resp = client.post("/classify")
        assert resp.status_code == 400
        assert b"No file part" in resp.data

    def test_empty_filename_returns_400(self, client):
        resp = client.post(
            "/classify",
            data={"file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_unsupported_extension_returns_400(self, client):
        resp = client.post(
            "/classify",
            data={"file": (io.BytesIO(b"data"), "file.bmp")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert b"Unsupported" in resp.data

    def test_valid_jpeg_returns_classification(self, client):
        img_bytes = _make_jpeg_bytes()
        with patch("app.classify_trash", return_value="Plastic"):
            resp = client.post(
                "/classify",
                data={"file": (io.BytesIO(img_bytes), "test.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        payload = json.loads(resp.data)
        assert payload["classification"] == "Plastic"

    def test_valid_png_is_accepted(self, client):
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), (0, 128, 255)).save(buf, format="PNG")
        buf.seek(0)
        with patch("app.classify_trash", return_value="Glass"):
            resp = client.post(
                "/classify",
                data={"file": (buf, "image.png")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200

    def test_classifier_runtime_error_returns_503(self, client):
        img_bytes = _make_jpeg_bytes()
        with patch("app.classify_trash", side_effect=RuntimeError("model down")):
            resp = client.post(
                "/classify",
                data={"file": (io.BytesIO(img_bytes), "test.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 503
        assert b"model down" in resp.data

    def test_upload_file_is_cleaned_up_after_success(self, client):
        """Uploaded file must be deleted even after a successful classification."""
        img_bytes = _make_jpeg_bytes()
        saved_path = None

        original_save = None

        def capture_save(path):
            nonlocal saved_path
            saved_path = path
            if original_save:
                original_save(path)

        with patch("app.classify_trash", return_value="Metal"):
            resp = client.post(
                "/classify",
                data={"file": (io.BytesIO(img_bytes), "capture.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        # The uploads dir should not contain the file after the request
        if saved_path:
            assert not os.path.exists(saved_path)

    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def limited_client():
    """Flask test client with rate limiter enabled."""
    with patch("classifier.TRASHNET_MODEL", None):
        with patch("classifier.load_trashnet_model", return_value=None):
            import importlib
            import app as app_module
            importlib.reload(app_module)
            app_module.app.config["TESTING"] = True
            app_module.limiter.enabled = True
            with app_module.app.test_client() as c:
                yield c


class TestRateLimiting:
    def test_returns_429_after_limit_exceeded(self, limited_client):
        img_bytes = _make_jpeg_bytes()

        with patch("app.classify_trash", return_value="Glass"):
            # 10 requests should succeed (limit is 10/minute)
            for _ in range(10):
                resp = limited_client.post(
                    "/classify",
                    data={"file": (io.BytesIO(img_bytes), "test.jpg")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200

            # 11th request must be rejected
            resp = limited_client.post(
                "/classify",
                data={"file": (io.BytesIO(img_bytes), "test.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 429
        payload = json.loads(resp.data)
        assert "Rate limit exceeded" in payload["error"]
