import logging
import os

from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename

from werkzeug.middleware.proxy_fix import ProxyFix

from classifier import TRASHNET_MODEL, classify_trash
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Trust one upstream proxy (Nginx) for X-Forwarded-* headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": f"Rate limit exceeded: {e.description}"}), 429


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
@limiter.limit("10 per minute;50 per hour")
def classify_image_route():
    """Receives an image, classifies it, and returns the result."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file extension"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            file.save(filepath)
            trash_class = classify_trash(filepath, TRASHNET_MODEL)
            logger.info("Classified '%s' as '%s'", filename, trash_class)
            return jsonify({"classification": trash_class})
        except RuntimeError as e:
            logger.warning("Classification unavailable for '%s': %s", filename, e)
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            logger.exception("Unexpected error classifying '%s'", filename)
            return jsonify({"error": "Error during classification"}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({"error": "An unknown error occurred"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
